"""PyInstaller entry point. Divert spawned workers before importing the CLI."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from hibiki_asr.standalone import main

    raise SystemExit(main())
