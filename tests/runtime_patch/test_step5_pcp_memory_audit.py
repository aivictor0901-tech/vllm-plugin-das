"""Shared views must not be reported as extra model allocations."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

PATH = Path(__file__).resolve().parents[2]/'vllm_hcu/v1/step5_pcp_memory_audit.py'
spec = importlib.util.spec_from_file_location('_memory_audit',PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def tensor(pointer, device='cuda:0', count=8, storage=64):
    return NS(device=device,dtype='int8',shape=(count,),numel=lambda:count,element_size=lambda:1,
              untyped_storage=lambda:NS(data_ptr=lambda:pointer,nbytes=lambda:storage))


class MemoryInventory(unittest.TestCase):
    def test_aliases_and_small_views_keep_one_full_storage(self):
        result = audit.storage_inventory([('weight',tensor(17,count=64)),
                                          ('alias',tensor(17,count=64)),
                                          ('small-view',tensor(17,count=1))])
        self.assertEqual(result['unique_storage_bytes'],64)
        self.assertEqual(result['logical_bytes_including_aliases'],129)
        self.assertEqual(result['largest_storages'][0]['alias_count'],3)

    def test_devices_and_separate_owners_remain_distinct_and_names_are_bounded(self):
        values = [('same-pointer-other-device',tensor(17,device='cuda:1'))]
        values += [('alias'+str(i),tensor(17)) for i in range(30)]
        values += [('different-owner',tensor(19))]
        result = audit.storage_inventory(values)
        self.assertEqual(result['unique_storage_bytes'],192)
        self.assertEqual(result['unique_storage_count'],3)
        self.assertTrue(all(len(row['names'])<=4 for row in result['largest_storages']))


if __name__ == '__main__':
    unittest.main()
