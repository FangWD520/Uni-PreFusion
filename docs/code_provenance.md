# Code provenance

This repository publishes the authors' Uni-PreFusion source snapshot on top of the [UniLION baseline](https://github.com/happinesslz/UniLION/tree/b75c28415ed53d5474bcb9be776d032c66ec432b). The original baseline commit is `b75c28415ed53d5474bcb9be776d032c66ec432b`; the fusion frontend and associated training/evaluation utilities are the supplied author modifications.

The public name is Uni-PreFusion. Existing model registration names, module paths, parameter keys and native-extension names are preserved for compatibility. Public configuration entry points and documentation use the new project name.

The release adds installation/data/training documentation, public configuration aliases, exact constructor-anchor resources, asset download/check tools, manuscript introduction and component attribution. It also corrects an inherited invalid relative-import statement in the unused KITTI evaluation entry point (`from . import kitti_common as kitti`). These packaging changes do not introduce a new model operator or numerical repair.

Training logs, datasets, compiled libraries, environment caches and later experimental runtime repairs are excluded. Checkpoints and Hilbert lookup tables are distributed separately with a SHA-256 manifest. The original model/checkpoint files are not rewritten to rename internal symbols.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for individual code origins and license limitations, and [reproducibility notes](reproducibility.md) for validation scope.
