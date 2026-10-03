"""Short end-to-end command checks with temporary outputs, no FFT."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
from cufhom.geometry.orientation import acg_closure


class A4CommandTests(unittest.TestCase):
    def invoke(self, output, *args):
        return subprocess.run([sys.executable, str(ROOT/'src/cufhom/cli/geometry.py'),
                               '--output', str(output), '--box', '12', '--ar', '2', '--vf', '.02',
                               '--grid', '36', '--initial-count', '12', '--candidates', '4',
                               '--threads', '1', *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=30)

    def test_explicit_a4_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            target = acg_closure(np.diag([1., 0., 0.]))
            source = folder/'orientation.json'
            source.write_text(json.dumps({'A4': target.tolist()}))
            result = self.invoke(folder/'out', '--orientation-json', str(source))
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            summary = json.loads((folder/'out/summary.json').read_text())
            self.assertTrue(summary['accepted'])
            self.assertEqual(summary['orientation_target']['source'], 'explicit_A4')
            self.assertTrue(summary['voxel_A4_ok'])
            with np.load(folder/'out/geometry.npz') as data:
                np.testing.assert_array_equal(data['A4_target'], target)

    def test_incompatible_a2_a4_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder/'orientation.json'
            source.write_text(json.dumps({'A4': acg_closure(np.diag([1., 0., 0.])).tolist()}))
            result = self.invoke(folder/'out', '--a2', '.5', '.3', '--orientation-json', str(source))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('inconsistent', result.stderr)
            self.assertFalse((folder/'out').exists())

    def test_a2_and_flexible_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            for model in ('straight', 'flexible'):
                extra = ['--solver', 'a2'] if model == 'straight' else ['--segments', '2']
                output = Path(temp)/model
                result = self.invoke(output, '--fiber-model', model, '--a2', '1', '0',
                                     '--geometry-only', *extra)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                request = json.loads((output/'request.json').read_text())
                self.assertEqual(request['settings']['solver'], 'a2')
                summary = json.loads((output/'summary.json').read_text())
                self.assertFalse(summary['accepted'])
                self.assertTrue(summary['vf_not_certified'])
                self.assertFalse((output/'geometry.npz').exists())

    def test_null_a4_does_not_silently_request_isotropy(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder/'orientation.json'
            source.write_text('{"A4": null}')
            result = self.invoke(folder/'out', '--orientation-json', str(source))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('cannot be null', result.stderr)
            self.assertFalse((folder/'out').exists())

    def test_a2_only_export_and_default_a4_closure_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            for solver in ('a2', 'auto'):
                output = Path(temp)/solver
                result = self.invoke(output, '--shape', 'cylinder', '--a2', '.6', '.3',
                                     '--solver', solver, '--overlap-tolerance', '1e-6')
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                summary = json.loads((output/'summary.json').read_text())
                self.assertTrue(summary['accepted'])
                with np.load(output/'geometry.npz', allow_pickle=False) as data:
                    np.testing.assert_allclose(data['A2_target'], np.diag([.6, .3, .1]))
                    if solver == 'a2':
                        self.assertEqual(data['A4_target'].size, 0)
                        self.assertNotIn('voxel_A4_error', summary)
                        self.assertIn('checking A2 only', result.stdout)
                    else:
                        self.assertEqual(data['A4_target'].shape, (3,3,3,3))
                        self.assertTrue(summary['voxel_A4_ok'])


if __name__ == '__main__':
    unittest.main()
