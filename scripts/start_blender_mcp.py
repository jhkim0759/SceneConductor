# Starter script for the BlenderMCP socket server.
#
# Invoked by a Claude Code hook as:
#   <blender_binary> --background --python ./scripts/start_blender_mcp.py
# so it runs inside Blender's own Python, in --background mode, from the repo root.
#
# Addon API used (discovered from blender_mcp_addon.py):
#   - module name        : "blender_mcp_addon"  (the addon file's __name__)
#   - port property      : bpy.context.scene.blendermcp_port (IntProperty, default 9876)
#   - start operator     : bpy.ops.blendermcp.start_server  (BLENDERMCP_OT_StartServer)
#                          -> creates bpy.types.blendermcp_server = BlenderMCPServer(port=...)
#                             and calls .start(), which binds the socket and spawns a
#                             DAEMON accept thread (BlenderMCPServer._server_loop).
#   - command pump       : module-level _blender_mcp_queue + _process_mcp_queue().
#                          The accept thread only *queues* commands; they are executed
#                          on Blender's main thread by _process_mcp_queue, which the
#                          addon normally drives via bpy.app.timers.register(...).
#
# Keep-alive choice:
#   Blender exits as soon as this script returns, which would tear down the socket
#   thread. We must block. In --background mode bpy.app.timers are NOT serviced
#   (there is no UI event loop), so the addon's registered _process_mcp_queue timer
#   never fires and queued commands would never run. Therefore the keep-alive loop
#   below both (a) blocks forever to keep the process/socket alive and (b) manually
#   drains the command queue by calling the addon's own _process_mcp_queue() each
#   tick -- exactly what the timer would have done in a foreground session.

import sys
import time
import traceback

import bpy
import addon_utils

ADDON_MODULE = "blender_mcp_addon"
PORT = 9876


def log(msg):
    print(f"[start_blender_mcp] {msg}", flush=True)


def main():
    log("Enabling BlenderMCP addon...")
    # enable() returns the loaded module object, which exposes the command queue
    # and the _process_mcp_queue pump we need to drive manually in background mode.
    addon_utils.enable(ADDON_MODULE, default_set=True, persistent=True)
    mod = sys.modules.get(ADDON_MODULE)
    if mod is None:
        raise RuntimeError(f"Addon module '{ADDON_MODULE}' failed to load")
    log(f"Addon '{ADDON_MODULE}' enabled")

    # Set the listening port on the scene before starting the server.
    scene = bpy.context.scene
    scene.blendermcp_port = PORT
    log(f"Configured blendermcp_port = {scene.blendermcp_port}")

    # Start the server via the addon's operator (creates BlenderMCPServer + accept thread).
    bpy.ops.blendermcp.start_server()
    log(f"BlenderMCP server listening on localhost:{PORT}")

    # Reference the addon's queue pump (None-safe: only used if present).
    pump = getattr(mod, "_process_mcp_queue", None)

    # Block forever, draining queued commands ourselves since timers don't tick
    # in --background mode. This keeps the process (and the daemon socket thread) alive.
    log("Entering keep-alive loop (Ctrl-C / kill to stop)")
    while True:
        if pump is not None:
            try:
                pump()
            except Exception as e:  # never let a single command kill the server
                log(f"Queue pump error: {e}")
                traceback.print_exc()
        time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted; shutting down")
    except Exception as e:
        log(f"FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
