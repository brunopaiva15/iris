# Publishing a model to the Hugging Face Hub

What to do, in order, the first time and every time after. The repository id
used throughout is `brunopaiva15/iris`; one repository holds the Iris family,
and each shipped version is a commit plus a tag (`v9`, `v10`…). That way a link
to the model keeps working, and a link to `v9` keeps meaning Iris 9 forever.

## Once, to set the account up

1. Create an account on [huggingface.co](https://huggingface.co) if there is
   none. The username becomes the namespace in `username/iris`.
2. Create a **write** access token: *Settings → Access Tokens → Create new
   token*, type **Write**, name it after the machine that will use it. A
   fine-grained token also works if it carries *Write access to contents* for
   this repository.
3. Install the client and log in:

   ```bash
   pip install -U "huggingface_hub[cli]"
   hf auth login        # paste the token
   hf auth whoami       # should print the username
   ```

   On a CI runner, skip the login and set `HF_TOKEN` in the environment
   instead.

## Every release

1. **Check the model against itself.** The publish script refuses to upload a
   model whose weights do not match the SHA-256 in `model.json`, whose
   `labels.txt` has the wrong number of lines, or whose card announces another
   version:

   ```bash
   python3 scripts/push_to_hub.py --model ../plant/assets/model --dry-run
   ```

   It should print the version, the class count, the size and `sha256 ok`.

2. **Update the model card.** `model-card/README.md` is the Hub page. Every
   number in it comes from `model.json`; nothing is typed by hand twice. At
   minimum, update the fact sheet, the results, and the version history.

3. **Upload.**

   ```bash
   python3 scripts/push_to_hub.py --model ../plant/assets/model --tag v9
   ```

   This creates the repository if it does not exist, uploads `plants.tflite`,
   `labels.txt` and `model.json`, publishes the card as `README.md`, and tags
   the commit. Files above 10 MB go through Git LFS automatically; at 9 MB the
   weights do not even need it.

4. **Look at the page.** `https://huggingface.co/brunopaiva15/iris`. Check that
   the licence badge reads *cc-by-4.0*, that the tags are there, that the
   metrics table rendered, and that the quick-start snippet is the current one.

5. **Make it public** — *Settings → Change repository visibility* — if it was
   created private. Everything before this step is reversible; this one is the
   one that publishes. A public repository can be made private again, but
   anything already downloaded is out.

## What not to publish

- **No training images.** They belong to the naturalists who took them. The
  model reproduces none of them, and that is the whole argument for training on
  CC BY-SA material in the first place.
- **No token in a commit.** `HF_TOKEN` lives in the environment, never in a
  file in the repository.
- **No metric that is not measured.** A number in the card that no script
  produced is a number that will be wrong at the next release and believed
  anyway.

## Deleting or renaming

A model repository can be renamed (*Settings → Rename*) and deleted
(*Settings → Delete this repository*), and both break every existing link and
every `hf_hub_download` call in the wild. Decide the name once. `iris` with
version tags is the shape that survives Iris 10 being a different architecture
altogether.
