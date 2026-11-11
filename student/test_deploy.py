"""Driver trial tests simulate hardware; they do not validate GPU compatibility."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import patch, MagicMock

spec = importlib.util.spec_from_file_location('course_deploy_setup', Path(__file__).parent / 'deploy/setup.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DriverTrialTests(unittest.TestCase):
    def setUp(self):
        self.profile = patch.object(deploy, 'SPEC', {'profile': '4090', 'cuda': '12.8', 'minimum_driver': '550.142', 'vllm': '0.10.2'})
        self.profile.start()
        self.addCleanup(self.profile.stop)

    def test_old_driver_still_rejected_by_default(self):
        with self.assertRaisesRegex(RuntimeError, '550.54.14'):
            deploy.driver_check('550.54.14', 'NVIDIA GeForce RTX 4090')

    def test_explicit_trial_records_bypass_not_validation(self):
        with patch.object(deploy, 'say'):
            status = deploy.driver_check('550.54.14', 'NVIDIA GeForce RTX 4090', True)
        self.assertTrue(status['driver_threshold_bypassed'])
        self.assertFalse(status['cuda_smoke_passed'])
        self.assertEqual(status['driver'], '550.54.14')

    def test_floor_and_gpu_restrictions(self):
        for driver, gpu in [('525.60.12', 'RTX 4090'), ('524.99', 'RTX 4090'), ('550.54.14', 'RTX 5090')]:
            with self.subTest(driver=driver, gpu=gpu), self.assertRaises(RuntimeError):
                deploy.driver_check(driver, gpu, True)
        with patch.object(deploy, 'say'):
            self.assertTrue(deploy.driver_check('525.60.13', 'RTX 4090', True)['driver_threshold_bypassed'])

    def test_normal_driver_needs_no_bypass(self):
        for driver in ('550.142', '550.142.0', '570.1'):
            self.assertFalse(deploy.driver_check(driver, 'RTX 4090', True)['driver_threshold_bypassed'])

    def test_cuda13_profile_cannot_use_trial(self):
        with patch.object(deploy, 'SPEC', {'profile': '5090', 'cuda': '13.0', 'minimum_driver': '580.65.06'}):
            with self.assertRaises(RuntimeError):
                deploy.driver_check('550.54.14', 'RTX 5090', True)

    def run_preflight(self, free_gib=50, compiler=True):
        runtime = MagicMock()
        runtime.exists.return_value = False
        (runtime / 'installed.json').is_file.return_value = False
        which = lambda name: '/bin/' + name if compiler or name not in ('gcc', 'cc') else None
        with patch.object(deploy.sys, 'platform', 'linux'), patch.object(deploy.platform, 'machine', return_value='x86_64'), \
             patch.object(deploy.shutil, 'which', side_effect=which), \
             patch.object(deploy.shutil, 'disk_usage', return_value=SimpleNamespace(free=free_gib * 1024**3)), \
             patch.object(deploy, 'RUNTIME', runtime), patch.object(deploy, 'say'), \
             patch.object(deploy, 'run', return_value=SimpleNamespace(stdout='550.54.14, NVIDIA GeForce RTX 4090')):
            return deploy.preflight(True)

    def test_preflight_allows_user_hardware(self):
        self.assertTrue(self.run_preflight()['driver_threshold_bypassed'])

    def test_trial_does_not_bypass_disk_or_compiler(self):
        with self.assertRaisesRegex(RuntimeError, '30 GiB'):
            self.run_preflight(free_gib=29)
        with self.assertRaisesRegex(RuntimeError, 'C编译器'):
            self.run_preflight(compiler=False)

    def test_cuda_failure_is_not_swallowed(self):
        with patch.object(deploy, 'run', side_effect=subprocess.CalledProcessError(1, 'python')):
            with self.assertRaises(subprocess.CalledProcessError):
                deploy.cuda_smoke_test('python')


if __name__ == '__main__':
    unittest.main()
