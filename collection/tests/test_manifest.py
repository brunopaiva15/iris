from plant_dataset.manifest import STATUS_KEPT, STATUS_REJECTED, ImageRecord, Manifest, now_iso, write_attributions


def record(**over):
    base = dict(
        species='Monstera deliciosa', internal_plant_id='monstera-deliciosa', source='gbif', source_id='1#0',
        original_url='https://www.inaturalist.org/observations/1', image_url='https://example.org/1.jpg', author='Ada',
        license='CC BY 4.0', license_url='https://creativecommons.org/licenses/by/4.0/', downloaded_at=now_iso(),
        checksum='a' * 64, path='Monstera_deliciosa/a.jpg', observation_id='1', phash='0' * 16,
    )
    base.update(over)
    return ImageRecord(**base)


def test_json_roundtrip_keeps_every_field():
    r = record(extra={'format': 'image/jpeg'})
    back = ImageRecord.from_json(r.to_json())
    assert back == r


def test_manifest_appends_and_reloads(tmp_path):
    m = Manifest(tmp_path / 'manifest.jsonl')
    m.append(record())
    m.append(record(source_id='2#0', checksum='b' * 64, observation_id='2'))
    again = Manifest(tmp_path / 'manifest.jsonl')
    assert len(again) == 2
    assert again.has_source('gbif', '1#0')
    assert not again.has_source('gbif', '9#0')
    assert again.by_checksum('b' * 64).source_id == '2#0'


def test_rewrite_after_status_change(tmp_path):
    m = Manifest(tmp_path / 'manifest.jsonl')
    m.append(record())
    m.records[0].status = STATUS_REJECTED
    m.records[0].reason = 'test'
    m.rewrite()
    again = Manifest(tmp_path / 'manifest.jsonl')
    assert again.records[0].status == STATUS_REJECTED
    assert again.records[0].reason == 'test'


def test_stats_count_by_species_status_license():
    m = Manifest.__new__(Manifest)
    m.records = [
        record(),
        record(source_id='2#0', checksum='b' * 64, license='CC0 1.0'),
        record(source_id='3#0', checksum='', status=STATUS_REJECTED, reason='trop petite'),
        record(species='Ficus elastica', internal_plant_id='ficus-elastica', source_id='4#0', checksum='c' * 64),
    ]
    s = m.stats()
    assert s['images'] == 4 and s['kept'] == 3
    assert s['species']['Monstera deliciosa'] == {'kept': 2, 'rejected': 1}
    assert s['licenses'] == {'CC BY 4.0': 2, 'CC0 1.0': 1}
    assert s['sources'] == {'gbif': 3}


def test_attributions_list_only_kept_images(tmp_path):
    records = [record(), record(source_id='3#0', checksum='', status=STATUS_REJECTED), record(species='Ficus elastica', author='Bob', source_id='4#0', checksum='c' * 64)]
    n = write_attributions(records, tmp_path / 'ATTRIBUTIONS.md', tmp_path / 'attributions.csv')
    assert n == 2
    md = (tmp_path / 'ATTRIBUTIONS.md').read_text()
    assert '## Ficus elastica' in md and 'Bob' in md and 'CC BY 4.0' in md
    csv_text = (tmp_path / 'attributions.csv').read_text()
    assert csv_text.count('\n') == 3  # header + 2 rows
    assert 'Ada' in csv_text and '3#0' not in csv_text
