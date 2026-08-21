from __future__ import annotations

import unittest
<<<<<<< HEAD
from pathlib import Path

from tools.build_with_debinfo import create_build_plan, debugify
=======

from tools.build_with_debinfo import (
    debugify,
    entry_command,
    extract_link_command,
    index_compile_commands,
)
>>>>>>> upstream/release/2.13


class TestDebugify(unittest.TestCase):
    def test_swaps_optimization_for_debug(self) -> None:
        self.assertEqual(debugify("cc -O2 -c a.cpp"), "cc -g -c a.cpp")
        self.assertEqual(debugify("cc -O3 -c a.cpp"), "cc -g -c a.cpp")

    def test_leaves_other_flags_untouched(self) -> None:
        cmd = "cc -DNDEBUG -I/x -fPIC -c a.cpp -o a.o"
        self.assertEqual(debugify(cmd), cmd)

    def test_metal_gets_debug_flags_once(self) -> None:
        out = debugify("xcrun metal -c a.metal")
        self.assertIn("-frecord-sources", out)
        self.assertIn("-gline-tables-only", out)
        # Idempotent: do not append a second time.
        self.assertEqual(out, debugify(out))


<<<<<<< HEAD
class TestCreateBuildPlan(unittest.TestCase):
    def test_follows_dependent_links(self) -> None:
        commands = "\n".join(
            [
                "c++ -O3 -o obj/other.o -c /repo/other.cpp",
                "c++ -O3 -o obj/a.o -c /repo/a.cpp",
                "c++ -shared -o lib/libtorch_cpu.so obj/a.o obj/other.o",
                "c++ -shared -o lib/libunrelated.so obj/other.o",
                "c++ -shared -o lib/libtorch_python.so lib/libtorch_cpu.so",
            ]
        )
        self.assertEqual(
            create_build_plan(["/repo/a.cpp"], commands, Path("/repo/build")),
            [
                ("debug rebuild", "c++ -g -o obj/a.o -c /repo/a.cpp"),
                (
                    "rebuild dependent",
                    "c++ -shared -o lib/libtorch_cpu.so obj/a.o obj/other.o",
                ),
                (
                    "rebuild dependent",
                    "c++ -shared -o lib/libtorch_python.so lib/libtorch_cpu.so",
                ),
            ],
        )

    def test_handles_metal_custom_commands(self) -> None:
        commands = "\n".join(
            [
                "cd /repo/build/metal && xcrun metal -c /repo/a.metal -o a.air",
                "cd /repo/build/metal && xcrun metallib -o a.metallib a.air",
                "c++ -shared -Wl,-sectcreate,/repo/build/metal/a.metallib -o lib/a.so",
            ]
        )
        plan = create_build_plan(["/repo/a.metal"], commands, Path("/repo/build"))
        self.assertEqual(len(plan), 3)
        self.assertIn("-frecord-sources -gline-tables-only", plan[0][1])

    def test_raises_when_source_is_absent(self) -> None:
        with self.assertRaises(RuntimeError):
            create_build_plan(
                ["/repo/missing.cpp"],
                "c++ -o obj/a.o -c /repo/a.cpp",
                Path("/repo/build"),
            )
=======
class TestEntryCommand(unittest.TestCase):
    def test_command_form(self) -> None:
        self.assertEqual(entry_command({"command": "cc -c a.cpp"}), "cc -c a.cpp")

    def test_arguments_form_is_quoted(self) -> None:
        entry = {"arguments": ["cc", "-c", "a b.cpp"]}
        self.assertEqual(entry_command(entry), "cc -c 'a b.cpp'")


class TestIndexCompileCommands(unittest.TestCase):
    def test_maps_resolved_source_paths(self) -> None:
        entries = [
            {"directory": "/repo/build", "file": "../torch/csrc/Module.cpp"},
            {"directory": "/repo/build", "file": "/repo/torch/csrc/Other.cpp"},
        ]
        index = index_compile_commands(entries)
        self.assertIn("/repo/torch/csrc/Module.cpp", index)
        self.assertIn("/repo/torch/csrc/Other.cpp", index)


class TestExtractLinkCommand(unittest.TestCase):
    def test_strips_ninja_wrapper(self) -> None:
        out = ": && clang++ -shared -o lib/libtorch_python.so a.o b.o && :"
        self.assertEqual(
            extract_link_command(out, "libtorch_python.so"),
            "clang++ -shared -o lib/libtorch_python.so a.o b.o",
        )

    def test_picks_link_among_compiles(self) -> None:
        out = "\n".join(
            [
                "clang++ -c torch_python.dir/Module.cpp.o",
                ": && clang++ -shared -o lib/libtorch_python.so a.o && :",
            ]
        )
        self.assertEqual(
            extract_link_command(out, "libtorch_python.so"),
            "clang++ -shared -o lib/libtorch_python.so a.o",
        )

    def test_raises_when_absent(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_link_command("clang++ -c a.o", "libtorch_python.so")
>>>>>>> upstream/release/2.13


if __name__ == "__main__":
    unittest.main()
