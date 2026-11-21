
import unittest
from pathlib import Path

from pipeline import (
    create_split_manifest,
    load_fixture,
    validate_records,
)


FIXTURE_PATH = Path(__file__).with_name("fixture.json")


class TestSplitPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.records = load_fixture(FIXTURE_PATH)

    def test_all_records_are_assigned_once(self):
        manifest = create_split_manifest(self.records)

        assigned = []
        for split in manifest["splits"].values():
            assigned.extend(split["record_ids"])

        expected = sorted(
            record["record_id"] for record in self.records
        )

        self.assertEqual(sorted(assigned), expected)
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_groups_do_not_overlap(self):
        manifest = create_split_manifest(self.records)

        group_sets = [
            set(split["group_ids"])
            for split in manifest["splits"].values()
        ]

        self.assertTrue(group_sets[0].isdisjoint(group_sets[1]))
        self.assertTrue(group_sets[0].isdisjoint(group_sets[2]))
        self.assertTrue(group_sets[1].isdisjoint(group_sets[2]))

    def test_same_seed_gives_same_manifest(self):
        first = create_split_manifest(self.records, seed=42)
        second = create_split_manifest(self.records, seed=42)

        self.assertEqual(first, second)

    def test_duplicate_record_ids_are_rejected(self):
        records = [dict(record) for record in self.records]
        records[1]["record_id"] = records[0]["record_id"]

        with self.assertRaises(ValueError):
            validate_records(records)

    def test_invalid_pixel_values_are_rejected(self):
        records = [dict(record) for record in self.records]
        records[0]["pixels"] = [[300, 0, 0, 0]] * 4

        with self.assertRaises(ValueError):
            validate_records(records)

    def test_fewer_than_three_groups_are_rejected(self):
        records = [dict(record) for record in self.records]
        records[2]["group_id"] = "group_A"
        records[3]["group_id"] = "group_A"
        records[4]["group_id"] = "group_B"
        records[5]["group_id"] = "group_B"

        with self.assertRaises(ValueError):
            create_split_manifest(records)


if __name__ == "__main__":
    unittest.main()

