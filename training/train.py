#!/usr/bin/env python3
"""Train the species classifier, then export it to TFLite.

    python3 train.py --dataset ../collection/dataset --out ./out

Transfer learning from a MobileNetV3 pretrained on ImageNet: the head is
replaced, trained alone for a few epochs with the rest frozen, then the top of
the network is unfrozen at a low learning rate. That is the recipe giving the
most accuracy per hour of computation when there are a few hundred images per
class.

The splits come from `splits.csv`: an observation cannot be in the training set
and in the test set at once, or the measured accuracy would be a lie.
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

IMAGE_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE
UNKNOWN = '_unknown'
# Reproducible shuffling: two runs start from the same order.
SHUFFLE_SEED = 20260905


def read_splits(dataset: Path) -> tuple[dict[str, list[tuple[str, str]]], set[str]]:
    """{split: [(absolute path, internal_id)]} as written by build_dataset, and
    the set of paths of cultivated-plant photographs (the `captive` column):
    accuracy on those is the only figure describing what the application will
    do on its users' photographs."""
    rows: dict[str, list[tuple[str, str]]] = {'train': [], 'val': [], 'test': []}
    captive: set[str] = set()
    with open(dataset / 'splits.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            path = dataset / row['path']
            if path.exists():
                rows[row['split']].append((str(path), row['internal_plant_id']))
                if row.get('captive') == '1':
                    captive.add(str(path))
    return rows, captive


def usable_classes(rows: dict, min_train: int, min_val: int) -> list[str]:
    """A class enters the model only if it has enough to learn from *and*
    enough to be evaluated on. A class with 3 images would produce an
    unreadable score."""
    train = Counter(pid for _, pid in rows['train'])
    val = Counter(pid for _, pid in rows['val'])
    return sorted(c for c in train if train[c] >= min_train and val[c] >= min_val)


LOAD_SIZE = 256    # a little margin around 224, for the crop
SOURCE_SIZE = 448  # the size the image set is stored at (../collection/plant_dataset/images.py, MAX_SIDE)


def set_input_size(px: int) -> None:
    """Change the network's input size, and the loading size with it.

    `LOAD_SIZE` keeps its crop margin — 256 for 224, one eighth. Changing one
    without the other is the trap of this option: the network would learn on a
    framing `model.json` does not announce, the application would feed it
    something other than what it saw, and the gap would be paid in points
    nobody could account for — which is exactly what cost the first version
    4.4 points.

    Beyond `SOURCE_SIZE`, we would be asking the set for more pixels than were
    stored: upscaling creates no detail, it only makes it look as though there
    were some.
    """
    global IMAGE_SIZE, LOAD_SIZE
    load = round(px * LOAD_SIZE / IMAGE_SIZE)
    if px < 32:
        raise SystemExit(f'--input-size {px}: too small')
    if load > SOURCE_SIZE:
        raise SystemExit(f'--input-size {px} asks for loading at {load} px, '
                         f'beyond the {SOURCE_SIZE} px the set is stored at')
    IMAGE_SIZE, LOAD_SIZE = px, load


def load_all(usable: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Decode into memory once and for all, as uint8 at the load size.

    Decoding a 1,024 px JPEG costs about 15 ms; redoing it every epoch would
    spend most of the training time inside the decoder. 9,000 images fit in
    1.8 GB — the right trade here.
    """
    images = np.zeros((len(usable), LOAD_SIZE, LOAD_SIZE, 3), dtype=np.uint8)
    labels = np.zeros(len(usable), dtype=np.int32)
    for i, (path, label) in enumerate(usable):
        raw = tf.io.read_file(path)
        img = tf.io.decode_jpeg(raw, channels=3)
        side = tf.reduce_min(tf.shape(img)[:2])
        img = tf.image.resize_with_crop_or_pad(img, side, side)
        images[i] = tf.image.resize(img, [LOAD_SIZE, LOAD_SIZE]).numpy().astype(np.uint8)
        labels[i] = label
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(usable)} images loaded', flush=True)
    return images, labels


def augment(image, label):
    """Random crop, mirror, and variation of light, colour and sharpness. No
    strong rotation: a pot stands upright in a photograph."""
    image = tf.image.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.random_brightness(image, 20.0)
    image = tf.image.random_saturation(image, 0.8, 1.25)
    image = resolution_jitter(image)
    return tf.clip_by_value(image, 0.0, 255.0), label


def resolution_jitter(image):
    """Shrink then re-enlarge the image, at a randomly drawn scale.

    The set was collected in three passes, with different storage sizes (1024,
    then 640, then 448 px). Resolution is therefore correlated with batches of
    species, and a network seizes that kind of shortcut before it learns any
    botany: recognising sharpness alone would let it eliminate two thirds of
    the classes. Blurring at random during training removes the option — and
    incidentally makes the model more robust to out-of-focus photographs,
    which are the everyday case.
    """
    def blurred():
        scale = tf.random.uniform([], 0.4, 1.0)
        small = tf.maximum(tf.cast(tf.cast(IMAGE_SIZE, tf.float32) * scale, tf.int32), 32)
        down = tf.image.resize(image, [small, small], method='bilinear')
        return tf.image.resize(down, [IMAGE_SIZE, IMAGE_SIZE], method='bilinear')

    return tf.cond(tf.random.uniform([]) < 0.5, blurred, lambda: image)


def center(image, label):
    offset = (LOAD_SIZE - IMAGE_SIZE) // 2
    image = tf.image.crop_to_bounding_box(image, offset, offset, IMAGE_SIZE, IMAGE_SIZE)
    return tf.cast(image, tf.float32), label


def array_dataset(images: np.ndarray, labels: np.ndarray, training: bool):
    """A dataset on top of the preloaded array, without copying it.

    `from_tensor_slices` on a numpy array makes it a constant of the graph:
    memory is doubled. On 25,000 images at 256 px that is 4.9 GB of array plus
    4.9 GB of copy, and the system kills the process. A generator reads the
    array in place; it permutes the order itself at every epoch, which mixes
    better than a sliding buffer.
    """
    n = len(labels)

    def generate():
        order = np.random.permutation(n) if training else np.arange(n)
        for i in order:
            yield images[i], labels[i]

    return tf.data.Dataset.from_generator(
        generate,
        output_signature=(
            tf.TensorSpec(shape=(LOAD_SIZE, LOAD_SIZE, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )


def read_and_square(path, label):
    """Read a JPEG and reduce it to the centred square at LOAD_SIZE, exactly as
    the in-memory preloading does. Both paths must produce the same image, or
    the measurements stop meaning anything."""
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    side = tf.reduce_min(tf.shape(image)[:2])
    image = tf.image.resize_with_crop_or_pad(image, side, side)
    image = tf.image.resize(image, [LOAD_SIZE, LOAD_SIZE])
    return tf.cast(image, tf.uint8), label


def make_dataset(pairs, classes: list[str], batch: int, training: bool, ram_budget_gb: float = 6.0, preload: bool = True, repeat: bool = False, skip: int = 0):
    """Preload into memory while it fits the budget, otherwise re-read the
    files every epoch.

    Preloading avoids decoding the same JPEGs sixteen times, but 30,000 images
    at 256 px are 5.6 GB: past the budget, a slower epoch beats a training run
    killed by the system.
    """
    index = {c: i for i, c in enumerate(classes)}
    usable = [(p, index[pid]) for p, pid in pairs if pid in index]
    if not usable:
        raise SystemExit('no usable image; has the dataset been built?')
    # splits.csv is sorted by species. Left in that order, a shuffle buffer of
    # a few thousand elements holds only a handful of species at a time: every
    # batch becomes nearly single-species, the model scores points by guessing
    # among ten classes, and generalises nothing — which is what was observed,
    # 31 % on training against 9 % on validation. So the whole list is shuffled
    # before tf.data sees it.
    random.Random(SHUFFLE_SEED).shuffle(usable)
    counts = Counter(label for _, label in usable)
    if skip:
        # Resuming an interrupted encoding. The shuffle is done, so the order
        # is the same as in the previous pass: cutting the already-computed
        # prefix is exactly resuming where it stopped.
        usable = usable[skip:]
    estimate = len(usable) * LOAD_SIZE * LOAD_SIZE * 3 / 1e9

    if preload and estimate <= ram_budget_gb:
        images, labels = load_all(usable)
        ds = array_dataset(images, labels, training)
        if training and repeat:
            ds = ds.repeat()
    else:
        if preload:
            print(f'  {len(usable)} images = {estimate:.1f} GB > budget {ram_budget_gb} GB: reading from files')
        paths = [p for p, _ in usable]
        labels = [l for _, l in usable]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            # The *paths* are shuffled, before decoding. Shuffling afterwards
            # meant reading eight thousand JPEGs before the first batch, seven
            # minutes at every epoch: invisible on a long epoch, ruinous on a
            # short one. Strings shuffle for free, and the whole list fits in
            # the buffer.
            ds = ds.shuffle(len(usable), reshuffle_each_iteration=True)
            if repeat:
                ds = ds.repeat()
        ds = ds.map(read_and_square, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(center, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE), counts, [p for p, _ in usable]


# Width of the vector returned by global average pooling, per network.
FEATURE_DIM = {'small': 576, 'large': 960}

BACKBONES = {
    'small': ('MobileNetV3Small', tf.keras.applications.MobileNetV3Small),
    'large': ('MobileNetV3Large', tf.keras.applications.MobileNetV3Large),
}


def usable_count(pairs, classes: list[str]) -> int:
    """How many images `make_dataset` will keep: those whose class made the
    cut. That count, not the one in `splits.csv`, says how many vectors the
    cache must hold."""
    index = set(classes)
    return sum(1 for _, pid in pairs if pid in index)


def frozen_backbone(backbone: str) -> tf.keras.Model:
    """The pretrained network, frozen, followed by its global average pooling:
    an image in, a vector out. This is exactly the part of `build_model` that
    does not move during the head phase."""
    _, factory = BACKBONES[backbone]
    base = factory(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet',
                   include_preprocessing=True, minimalistic=False)
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image')
    x = base(inputs, training=False)
    return tf.keras.Model(inputs, tf.keras.layers.GlobalAveragePooling2D()(x))


def encode(pairs, classes: list[str], batch: int, backbone: str, ram_budget_gb: float,
           label: str, x_path: Path | None = None, y_path: Path | None = None,
           mark=None, done: int = 0):
    """Push the images through the frozen network once and for all.

    During the head phase the network does not move: re-encoding the same
    images every epoch is doing the same computation four times. One forward
    pass produces one vector per image; the head then trains on those vectors,
    in seconds instead of half an hour per epoch.

    The vectors are written as they come, not at the end. This pass takes over
    an hour on 232,000 images and the machine can disappear meanwhile: keeping
    everything in memory until the end means losing everything. Resuming is
    exact to the vector.

    The price is the framing: the vectors are those of the centred square,
    without augmentation. For a linear head whose only job is to start
    somewhere other than random before fine-tuning, that is inconsequential —
    and fine-tuning keeps all of its augmentations.
    """
    ds, counts, paths = make_dataset(pairs, classes, batch, training=False,
                                     ram_budget_gb=ram_budget_gb, preload=False, skip=done)
    model = frozen_backbone(backbone)
    total = done + len(paths)
    chunks, labels, at, t0 = [], [], done, time.time()
    x = np.lib.format.open_memmap(x_path, mode='r+' if done else 'w+', dtype=np.float16,
                                  shape=(total, FEATURE_DIM[backbone])) if x_path else None
    y = np.lib.format.open_memmap(y_path, mode='r+' if done else 'w+', dtype=np.int32,
                                  shape=(total,)) if y_path else None
    for images, batch_labels in ds:
        vectors = model(images, training=False).numpy().astype(np.float16)
        n = int(batch_labels.shape[0])
        if x is not None:
            x[at:at + n] = vectors
            y[at:at + n] = batch_labels.numpy()
        else:
            chunks.append(vectors)
            labels.append(batch_labels.numpy())
        at += n
        if (at - done) % (batch * 100) < batch:
            if x is not None:
                x.flush()
                y.flush()
                if mark:
                    mark(at)
            rate = (at - done) / max(1e-6, time.time() - t0)
            print(f'  {label}: {at}/{total} encoded, {rate:.0f} img/s, '
                  f'{(total - at) / rate / 60:.0f} min left', flush=True)
    if x is not None:
        x.flush()
        y.flush()
        if mark:
            mark(at)
        return x, y, counts
    return np.concatenate(chunks), np.concatenate(labels), counts


def encoded(cache: Path | None, name: str, signature: dict, encode_from):
    """Return the vectors, resuming the encoding where it stopped.

    The signature file says on which set and which network the vectors were
    computed, and how many are written. A signature that no longer matches
    starts over: an hour of computation beats a head trained on another set's
    vectors.
    """
    if cache is None:
        x, y, _ = encode_from(None, None, None, 0)
        return x, y
    cache.mkdir(parents=True, exist_ok=True)
    meta_p, x_p, y_p = cache / f'{name}.json', cache / f'{name}.x.npy', cache / f'{name}.y.npy'
    done, total = 0, signature['images']
    if meta_p.exists() and x_p.exists() and y_p.exists():
        meta = json.loads(meta_p.read_text())
        if {k: v for k, v in meta.items() if k != 'done'} == signature:
            done = int(meta.get('done', 0))
            if done >= total:
                print(f'  {name}: vectors read back from the cache')
                return np.load(x_p, mmap_mode='r'), np.load(y_p, mmap_mode='r')
            print(f'  {name}: resuming at {done}/{total} vectors')
        else:
            print(f'  {name}: stale cache, re-encoding')

    def mark(at):
        meta_p.write_text(json.dumps({**signature, 'done': at}))

    x, y, _ = encode_from(x_p, y_p, mark, done)
    return x, y


def fit_head(x, y, val, n_classes: int, dropout: float, epochs: int, weights: dict, batch: int):
    """The head alone, on the precomputed vectors. Returns its weights, to be
    put back into the full model before fine-tuning."""
    head = tf.keras.Sequential([
        tf.keras.Input(shape=(x.shape[1],)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_classes, activation='softmax', name='species'),
    ])
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    head.fit(x.astype(np.float32), y, validation_data=(val[0].astype(np.float32), val[1]),
             epochs=epochs, batch_size=batch * 8, class_weight=weights, verbose=2,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3,
                                                         restore_best_weights=True)])
    return head.get_layer('species').get_weights()



def use_mixed_precision() -> str:
    """Compute in float16 where it is safe, accumulate in float32.

    On a card with tensor cores (RTX 20xx and later) this roughly doubles
    throughput and halves memory, which allows bigger batches. On a CPU it
    gains nothing and can be slower: an option, not a default.

    The weights stay in float32; only intermediate computations move to
    float16. The head is already forced to float32 (`build_model`), and the
    TFLite export starts from the weights rather than from the compute policy:
    the shipped file is the same.
    """
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    return tf.keras.mixed_precision.global_policy().name


def describe_devices() -> str:
    """What the training will actually run on.

    An invisible card — missing driver, TensorFlow without CUDA, WSL set up
    wrong — shows up as a training run ten times slower, and as no message at
    all. Better said at startup."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return 'no GPU visible: training on the CPU'
    for gpu in gpus:
        # Without this TensorFlow reserves the whole card's memory at startup,
        # and nothing else can use it any more.
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return f'{len(gpus)} GPU(s): ' + ', '.join(g.name for g in gpus)


def build_model(n_classes: int, dropout: float, backbone: str = 'small') -> tf.keras.Model:
    """The network: an ImageNet-pretrained MobileNetV3 without its head, then
    ours. `small` (2.5 MB in float16, ~15 ms on a recent phone) served up to
    the third version; `large` (~11 MB, three times the computation) sees more
    of the detail that separates two close species."""
    name, factory = BACKBONES[backbone]
    base = factory(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet',
        include_preprocessing=True, minimalistic=False)
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image')
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # The last layer stays in float32, even under mixed precision: a float16
    # softmax overflows as soon as the logits pass ~11, and the probabilities
    # it returns are what the application's threshold reads.
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='species', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)
    model.base = base
    model.architecture = name
    return model


def class_weights(counts: Counter, n_classes: int) -> dict[int, float]:
    """Rare classes count for more: without this, the model learns to answer
    "Monstera" and is right one time in ten."""
    total = sum(counts.values())
    return {i: total / (n_classes * max(1, counts.get(i, 0))) for i in range(n_classes)}


def evaluate(model, ds, classes: list[str], captive_mask=None) -> dict:
    """Top-1, top-3, macro-F1, and the threshold / fallback curve used to tune
    the acceptance rule on the application side.

    **None of this needs the full matrix of probabilities.** An earlier version
    kept it, then had `np.argsort` sort it across its whole width: 99,825
    images x 5,376 classes are 2.1 GB of floats, the sort made a copy in 64-bit
    integers twice that size, and the peak passed nine gigabytes — in order to
    read three columns afterwards. The kernel's out-of-memory killer took the
    evaluation of a model after eight hours of training, at the very last step.

    So, batch by batch, only **the three best indices and the two best
    probabilities** are kept: a few megabytes instead of nine gigabytes, and
    the same numbers to the last decimal.
    """
    tops, bests, truth = [], [], []
    for images, labels in ds:
        p = model.predict(images, verbose=0)
        # `argpartition` brings the three best to the front without sorting the
        # rest — the sort was the expensive part. Only those three are then
        # sorted, to put them in the expected descending order.
        three = np.argpartition(-p, min(3, p.shape[1] - 1), axis=1)[:, :3]
        rank = np.argsort(-np.take_along_axis(p, three, axis=1), axis=1)
        order = np.take_along_axis(three, rank, axis=1)
        tops.append(order.astype(np.int32))
        bests.append(np.take_along_axis(p, order[:, :2], axis=1).astype(np.float32))
        truth.append(labels.numpy())
    if not tops:
        return {}
    order = np.concatenate(tops)
    best = np.concatenate(bests)
    truth = np.concatenate(truth)
    top1 = order[:, 0]
    correct = top1 == truth
    top3 = np.mean([t in o for t, o in zip(truth, order)])

    f1s = []
    for c in range(len(classes)):
        tp = int(np.sum((top1 == c) & (truth == c)))
        fp = int(np.sum((top1 == c) & (truth != c)))
        fn = int(np.sum((top1 != c) & (truth == c)))
        if tp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    margin = best[:, 0] - best[:, 1]

    # The number that matters for the application: on the photographs of
    # cultivated plants only, in pots, in people's homes. The rest of the set
    # is wild plants, which nobody photographs in a living room.
    captive_metrics = None
    if captive_mask is not None:
        mask = np.asarray(list(captive_mask), dtype=bool)[:len(truth)]
        if int(mask.sum()) > 0:
            c_top3 = np.mean([t in o for t, o in zip(truth[mask], order[mask])])
            c_rows = []
            for threshold in (0.5, 0.7, 0.9):
                acc = (best[mask][:, 0] >= threshold)
                n = int(acc.sum())
                c_rows.append({'threshold': threshold, 'accepted_rate': round(n / int(mask.sum()), 4),
                               'precision_when_accepted': round(float(np.mean(correct[mask][acc])), 4) if n else None})
            captive_metrics = {'images': int(mask.sum()), 'top1': round(float(np.mean(correct[mask])), 4),
                               'top3': round(float(c_top3), 4), 'threshold_curve': c_rows}
    curve = []
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        for min_margin in (0.0, 0.15, 0.25, 0.4):
            accepted = (best[:, 0] >= threshold) & (margin >= min_margin)
            n = int(np.sum(accepted))
            curve.append({
                'threshold': threshold, 'min_margin': min_margin,
                'accepted_rate': round(n / len(truth), 4),
                'precision_when_accepted': round(float(np.mean(correct[accepted])), 4) if n else None,
            })
    return {
        'images': int(len(truth)), 'top1': round(float(np.mean(correct)), 4), 'top3': round(float(top3), 4),
        'macro_f1': round(float(np.mean(f1s)), 4) if f1s else None,
        'mean_confidence': round(float(np.mean(best[:, 0])), 4),
        'threshold_curve': curve,
        'captive': captive_metrics,
    }


def export_tflite(model, out: Path, classes: list[str], names: dict, metrics: dict, quantize_ds=None,
                  masks: dict[str, list[str]] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    blob = converter.convert()
    model_path = out / 'plants.tflite'
    model_path.write_bytes(blob)
    (out / 'labels.txt').write_text('\n'.join(classes) + '\n', encoding='utf-8')
    meta = {
        'version': metrics.get('version', '1'),
        'input_size': IMAGE_SIZE,
        # The preprocessing recipe, so that the application applies exactly the
        # same one: resize the centred square to `load_size`, then centre crop
        # to `input_size`. Resizing straight to the input size changes the
        # framing and costs several points of accuracy.
        'load_size': LOAD_SIZE,
        # The size the set's images were reduced to for storage. The
        # application repeats the same two-step reduction on the device's
        # photographs; without it, it would feed the model images more aliased
        # than anything it has ever seen.
        'source_size': SOURCE_SIZE,
        'classes': len(classes),
        'architecture': getattr(model, 'architecture', 'MobileNetV3Small'),
        'preprocessing': 'included_in_graph_uint8_0_255',
        'sha256': hashlib.sha256(blob).hexdigest(),
        'bytes': len(blob),
        'metrics': {k: v for k, v in metrics.items() if k != 'threshold_curve'},
        'threshold_curve': metrics.get('threshold_curve', []),
        'species': {c: names.get(c, c) for c in classes},
    }
    if masks:
        # The context masks. A union model carries all its classes in
        # `labels.txt` and says here which belong to which place; the
        # application renormalises over the classes of the place —
        # `exp(zi) / sum_kept` — which yields exactly what this model resliced
        # on that mask would yield. A single-domain model writes nothing:
        # without this object, the application does not mask.
        kept = set(classes)
        cleaned = {name: [c for c in classes if c in set(ids) & kept] for name, ids in masks.items()}
        meta['masks'] = {name: ids for name, ids in cleaned.items() if ids and len(ids) < len(classes)}
    (out / 'model.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    return meta


def species_names(dataset: Path) -> dict:
    """internal_id -> scientific name, read from the manifest."""
    names = {}
    with open(dataset / 'manifest.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            names.setdefault(r['internal_plant_id'], r['species'])
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='../collection/dataset')
    ap.add_argument('--out', default='./out')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--head-epochs', type=int, default=4)
    ap.add_argument('--fine-epochs', type=int, default=12)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--min-train', type=int, default=25, help='minimum training images per class')
    ap.add_argument('--min-val', type=int, default=3)
    ap.add_argument('--unfreeze', type=int, default=60, help='layers unfrozen at the top of the network')
    ap.add_argument('--fine-lr', type=float, default=5e-5, help='fine-tuning learning rate')
    ap.add_argument('--version', default='1')
    ap.add_argument('--backbone', choices=sorted(BACKBONES), default='small', help='MobileNetV3 small or large')
    ap.add_argument('--input-size', type=int, default=IMAGE_SIZE,
                    help='side of the network input, in pixels; the loading size follows at the '
                         'same crop margin. 320 is the classic lever of fine-grained recognition, '
                         'at the price of inference twice as heavy on the phone')
    ap.add_argument('--ram-budget', type=float, default=5.0, help='GB of preloading at most; beyond that, read from the files')
    ap.add_argument('--steps-per-epoch', type=int, help='batches per epoch; a short epoch means frequent checkpoints')
    ap.add_argument('--val-max', type=int, default=6000, help='validation images during training; the final evaluation stays complete')
    ap.add_argument('--checkpoint', help='folder where the weights are saved after each epoch, and from which training resumes')
    ap.add_argument('--feature-cache', help='folder holding the frozen network activations; the head phase becomes one forward pass instead of N epochs')
    ap.add_argument('--mixed-precision', action='store_true', help='compute in float16: doubles throughput on a card with tensor cores, useless on a CPU')
    args = ap.parse_args()
    # Before any reading of the set or building of the network: everything else
    # in this file reads these constants at the moment it uses them.
    if args.input_size != IMAGE_SIZE:
        set_input_size(args.input_size)

    print(describe_devices(), flush=True)
    if args.mixed_precision:
        print(f'mixed precision: {use_mixed_precision()}', flush=True)

    dataset = Path(args.dataset)
    rows, captive = read_splits(dataset)
    classes = usable_classes(rows, args.min_train, args.min_val)
    if len(classes) < 2:
        raise SystemExit(f'{len(classes)} usable class(es): the collection is too thin')
    names = species_names(dataset)
    print(f'{len(classes)} classes, {len(rows["train"])} train / {len(rows["val"])} val / {len(rows["test"])} test')

    train_ds, counts, _ = make_dataset(rows['train'], classes, args.batch, training=True, ram_budget_gb=args.ram_budget,
                                       repeat=bool(args.steps_per_epoch))
    # Per-epoch validation runs on a sample: it is there to follow the curve
    # and to decide when to stop, not to measure the model. The measurement is
    # the test set at the end, whole. Both are read from the files: preloading
    # 47,000 JPEGs cost ten minutes at startup, and this machine restarts
    # often.
    val_rows = rows['val']
    if args.val_max and len(val_rows) > args.val_max:
        val_rows = random.Random(SHUFFLE_SEED).sample(val_rows, args.val_max)
    val_ds, _, _ = make_dataset(val_rows, classes, args.batch, training=False, ram_budget_gb=args.ram_budget, preload=False)

    model = build_model(len(classes), args.dropout, args.backbone)
    weights = class_weights(counts, len(classes))

    # Checkpoints. This machine can disappear at any moment and an epoch costs
    # half an hour: the weights are therefore written during an epoch as well,
    # not only at its end. Only the end advances the epoch counter; resuming
    # mid-epoch amounts to redoing it from slightly more advanced weights,
    # which costs nothing.
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    head_w = ckpt / 'head.weights.h5' if ckpt else None
    fine_w = ckpt / 'fine.weights.h5' if ckpt else None
    state_p = ckpt / 'state.json' if ckpt else None
    state = json.loads(state_p.read_text()) if state_p and state_p.exists() else {}
    head_done = state.get('head_epochs_done', 0)
    fine_done = state.get('fine_epochs_done', 0)
    if ckpt:
        ckpt.mkdir(parents=True, exist_ok=True)
        # The class list, written **before** the hours of computation.
        # `labels.txt` only exists at export time: an evaluation that falls
        # over therefore takes with it the correspondence between the head's
        # columns and the species, and weights without that list are of no use
        # at all. It once had to be rebuilt by hand after eight hours of
        # training. A few kilobytes of insurance.
        (ckpt / 'classes.txt').write_text('\n'.join(classes) + '\n', encoding='utf-8')

    def _save_state(**kw):
        state.update(kw)
        state_p.write_text(json.dumps(state))

    class _Checkpoint(tf.keras.callbacks.Callback):
        def __init__(self, path, key, every=200):
            super().__init__()
            self.path, self.key, self.every = path, key, every
            self.t0 = time.time()

        def on_train_batch_end(self, batch, logs=None):
            if batch and batch % self.every == 0:
                self.model.save_weights(self.path)
                # A heartbeat: on a machine that restarts, this is the only way
                # to know whether training is really progressing.
                print(f'  batch {batch} saved, {time.time() - self.t0:.0f} s', flush=True)

        def on_epoch_end(self, epoch, logs=None):
            self.model.save_weights(self.path)
            _save_state(**{self.key: epoch + 1, 'val_accuracy': float((logs or {}).get('val_accuracy', 0))})

    in_fine = bool(fine_w and fine_w.exists())
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if in_fine:
        print(f'resuming: fine-tuning, {fine_done} epoch(s) done')
    elif head_w and head_w.exists():
        model.load_weights(head_w)
        print(f'resuming: head, {head_done} epoch(s) done')
    if not in_fine and head_done < args.head_epochs:
        if args.feature_cache:
            # The network is frozen: the vectors it produces do not change from
            # one epoch to the next. They are computed once, then the head
            # trains on them in minutes instead of half an hour per epoch — and
            # can run to convergence, which gives fine-tuning a far better
            # starting point than four epochs.
            cache = Path(args.feature_cache)
            signature = {'backbone': args.backbone, 'input': IMAGE_SIZE, 'classes': len(classes),
                         'fingerprint': hashlib.sha256('\n'.join(classes).encode()).hexdigest()[:16]}
            def encoder(pairs, label):
                def run(x_p, y_p, mark, done):
                    return encode(pairs, classes, args.batch, args.backbone, args.ram_budget,
                                  label, x_p, y_p, mark, done)
                return run

            x, y = encoded(cache, 'train', {**signature, 'images': usable_count(rows['train'], classes)},
                           encoder(rows['train'], 'training'))
            xv, yv = encoded(cache, 'val', {**signature, 'images': usable_count(val_rows, classes)},
                             encoder(val_rows, 'validation'))
            print(f'head on vectors: {x.shape[0]} x {x.shape[1]}, validation {xv.shape[0]}')
            model.get_layer('species').set_weights(
                fit_head(x, y, (xv, yv), len(classes), args.dropout, args.head_epochs, weights, args.batch))
            del x, y, xv, yv
            if ckpt:
                model.save_weights(head_w)
                _save_state(head_epochs_done=args.head_epochs)
        else:
            model.fit(train_ds, validation_data=val_ds, initial_epoch=head_done, epochs=args.head_epochs,
                      steps_per_epoch=args.steps_per_epoch, class_weight=weights, verbose=2,
                      callbacks=[_Checkpoint(head_w, 'head_epochs_done')] if ckpt else [])

    model.base.trainable = True
    for layer in model.base.layers[:-args.unfreeze]:
        layer.trainable = False
    # The batch-normalisation layers stay frozen: unfrozen, they recompute
    # their statistics on batches of 32 images and destroy in one epoch what
    # ImageNet pretraining had established. That is the classic cause of a
    # validation score collapsing at the start of fine-tuning.
    frozen_bn = 0
    for layer in model.base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1
    print(f'fine-tuning: {args.unfreeze} layers unfrozen, {frozen_bn} normalisations frozen')
    model.compile(optimizer=tf.keras.optimizers.Adam(args.fine_lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if in_fine:
        model.load_weights(fine_w)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)]
    if ckpt:
        callbacks.append(_Checkpoint(fine_w, 'fine_epochs_done'))
    if fine_done < args.fine_epochs:
        model.fit(train_ds, validation_data=val_ds, initial_epoch=fine_done, epochs=args.fine_epochs,
                  steps_per_epoch=args.steps_per_epoch, class_weight=weights, verbose=2, callbacks=callbacks)

    if args.mixed_precision:
        # The TFLite converter cannot convert a float16 graph: it asks for
        # "flex ops", which the application does not embed. The weights, for
        # their part, stayed in float32 — mixed precision only changes the
        # intermediate computations. So the network is rebuilt in float32 and
        # the learned weights are put back into it.
        #
        # The switch happens *before* the evaluation, not only before the
        # export: that way the numbers published in `model.json` are those of
        # the shipped file, and not those of a model resembling it.
        tf.keras.mixed_precision.set_global_policy('float32')
        weights = model.get_weights()
        model = build_model(len(classes), args.dropout, args.backbone)
        model.set_weights(weights)
        print('mixed precision: network rebuilt in float32 for evaluation and export', flush=True)

    test_ds, _, test_paths = make_dataset(rows['test'], classes, args.batch, training=False,
                                          ram_budget_gb=args.ram_budget, preload=False)
    metrics = evaluate(model, test_ds, classes, captive_mask=[p in captive for p in test_paths])
    metrics['version'] = args.version
    print(json.dumps({k: v for k, v in metrics.items() if k != 'threshold_curve'}, indent=1))
    if metrics.get('captive'):
        c = metrics['captive']
        print(f"cultivated plants ({c['images']} test images): top1 {c['top1']}, top3 {c['top3']}")

    meta = export_tflite(model, Path(args.out), classes, names, metrics)
    print(f'model written: {args.out}/plants.tflite — {meta["bytes"] / 1e6:.1f} MB, {meta["classes"]} classes')
    for row in metrics.get('threshold_curve', []):
        if row['min_margin'] == 0.25 and row['threshold'] in (0.8, 0.9):
            print(f'  threshold {row["threshold"]} margin 0.25 → {row["accepted_rate"]:.0%} accepted, '
                  f'precision {row["precision_when_accepted"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
