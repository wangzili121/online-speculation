"""Test harness: hold 16 GiB during compilation, hand it back as model allocation starts."""
import ctypes
import json
import os
from pathlib import Path
import runpy
import select
import subprocess
import sys
import threading

MARK = 'ORTHRUS_GUARD_JSON '


def hold():
    acl = ctypes.CDLL(str(Path(os.environ["ASCEND_HOME_PATH"])/"lib64/libascendcl.so"))
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclrtSetDevice.argtypes = [ctypes.c_int]
    acl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    acl.aclrtFree.argtypes = [ctypes.c_void_p]
    def check(code):
        if code != 0:
            raise RuntimeError('guard ACL return '+str(code))
    def send(value):
        print(MARK+json.dumps(value), flush=True)
    values = {}
    try:
        check(acl.aclInit(None))
        for device in range(len(os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(','))):
            check(acl.aclrtSetDevice(device))
            values[device] = []
            for _ in range(16):
                ptr = ctypes.c_void_p()
                check(acl.aclrtMalloc(ctypes.byref(ptr), 1 << 30, 0))
                values[device].append(ptr)
        send(dict(ok=True, reserved_gib=16, devices=list(values)))
        for line in sys.stdin:
            command = json.loads(line)
            devices = list(values) if command['op'] in {'all', 'quit'} else [command['device']]
            for device in devices:
                check(acl.aclrtSetDevice(device))
                while values[device]:
                    check(acl.aclrtFree(values[device][-1]))
                    values[device].pop()
            send(dict(ok=True, released=devices))
            if command['op'] == 'quit':
                break
    except BaseException as error:
        send(dict(ok=False, error=repr(error)))
        raise
    finally:
        for device, pointers in values.items():
            acl.aclrtSetDevice(device)
            for ptr in pointers:
                acl.aclrtFree(ptr)
        acl.aclFinalize()


class Guard:
    def __init__(self):
        self.process = subprocess.Popen([sys.executable, __file__, '--hold'], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, bufsize=0)
        self.closed = False
        self.devices = set(range(len(os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(","))))
        self.lock = threading.Lock()
        self.released, self.allocated, self.pending = set(), {}, b''
        try:
            self.read()
        except BaseException:
            self._reap()
            raise
    def read(self):
        while True:
            while b'\n' not in self.pending:
                if not select.select([self.process.stdout], [], [], 30)[0]:
                    raise TimeoutError('guard response')
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError('guard terminated')
                self.pending += data
            raw, self.pending = self.pending.split(b'\n', 1)
            line = raw.decode(errors='replace')+'\n'
            if line.startswith(MARK):
                value = json.loads(line[len(MARK):])
                if not value['ok']:
                    raise RuntimeError(value)
                print('guard', value, flush=True)
                return value
            print(line, end='', flush=True)
    def release(self, device=None):
        with self.lock:
            if device is not None and device in self.released:
                return
            self.process.stdin.write((json.dumps(dict(op='all' if device is None else 'device', device=device))+'\n').encode())
            self.process.stdin.flush()
            self.released.update(self.read()['released'])
    def allocate(self, device, size):
        if self.process.poll() is not None:
            return
        self.allocated[device] = self.allocated.get(device, 0)+size
        if self.allocated[device] > 32 << 30:
            self.release(device)
    def _reap(self):
        if not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # 仅终止本对象创建的保留显存子进程，不重置设备或接触其他进程。
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            print('guard', dict(shutdown='terminated owned reservation helper', pid=self.process.pid), flush=True)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                if self.process.poll() is None:
                    self.process.stdin.write(b'{"op":"quit"}\n'); self.process.stdin.flush()
                    self.released.update(self.read()['released'])
                    if not self.devices <= self.released:
                        raise RuntimeError('guard did not acknowledge all allocation releases')
            finally:
                self._reap()


def main():
    if sys.argv[1:] == ['--hold']:
        hold(); return
    script, args = sys.argv[1], sys.argv[2:]
    guard = Guard()
    stop = threading.Event()
    try:
        from pypto.runtime import DistributedWorker
        original_init, original_malloc = DistributedWorker.__init__, DistributedWorker.malloc
        def initialize(self, *args, config=None, **kwargs):
            self._guard_devices = tuple(config.distributed_config.device_ids)
            return original_init(self, *args, config=config, **kwargs)
        def malloc(self, nbytes, *, worker_id=0):
            guard.allocate(self._guard_devices[worker_id], nbytes)
            return original_malloc(self, nbytes, worker_id=worker_id)
        DistributedWorker.__init__, DistributedWorker.malloc = initialize, malloc
        from blockspec.pypto_qwen38.target import OrthrusRunner
        original_start = OrthrusRunner.start
        def start(self):
            result = original_start(self)
            guard.release()
            guard.close()
            return result
        OrthrusRunner.start = start
        if Path(script).name == 'smoke_qwen38_pypto_base.py':
            log = Path(args[args.index('--output-dir')+1])/'server.log'
            def watch():
                offset, tail = 0, ''
                while not stop.wait(.1):
                    if log.exists():
                        with log.open(errors='replace') as file:
                            file.seek(offset); data=file.read(); offset=file.tell()
                        tail += data
                        if 'loaded target bank on device mlp_up' in tail:
                            guard.release(); return
                        tail = tail[-200:]
            threading.Thread(target=watch, daemon=True).start()
        sys.path.insert(0, str(Path(script).parent))
        sys.argv = [script, *args]
        runpy.run_path(script, run_name='__main__')
    finally:
        stop.set()
        guard.close()


if __name__ == '__main__':
    main()
