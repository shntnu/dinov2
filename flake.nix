{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
      with pkgs;
      rec {
        apps.default =
          let
            python_with_pkgs = python3.withPackages (pp: [
              (inputs.nahual-flake.packages.${system}.nahual)
              packages.dinov2
            ]);
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/dinov2.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };
        formatter = pkgs.alejandra;
        packages = pkgs.callPackage ./nix { };
        devShells = {
          default =
            let
              python_with_pkgs = (
                python3.withPackages (pp: [
                  (inputs.nahual-flake.packages.${system}.nahual)
                  packages.dinov2
                ])
              );
            in
            mkShell {
              packages = [
                python_with_pkgs
              ];
              currentSystem = system;
              venvDir = "./.venv";
              postVenvCreation = ''
                unset SOURCE_DATE_EPOCH
              '';
              postShellHook = ''
                unset SOURCE_DATE_EPOCH
              '';
              shellHook = ''
                export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
                export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
                export NVCC_APPEND_FLAGS="-Xcompiler -fno-PIC"
                export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
                export CUDA_NVCC_FLAGS="-O2 -Xcompiler -fno-PIC"
                runHook venvShellHook
                # PYTHONSAFEPATH=1 (Python 3.11+) keeps Python from prepending
                # the script's directory (or cwd for python -c mode) to
                # sys.path, which would otherwise let the in-tree dinov2/
                # source dir shadow the nix-built package.
                export PYTHONSAFEPATH=1
                export PYTHONDONTWRITEBYTECODE=1
              '';
            };

          # Minimal shell for the pixi-based path. Exposes pixi and the system
          # NVIDIA driver libs so conda-installed pytorch-gpu can load libcuda
          # on NixOS, where /run/opengl-driver/lib is not on a default search
          # path. Use as:  nix develop .#pixi --command pixi run dinov2 ...
          # (Pattern from shntnu/neusis templates/python-pixi.)
          pixi = mkShell {
            packages = [ pkgs.pixi ];
            LD_LIBRARY_PATH = "/run/opengl-driver/lib";
          };
        };
      }
    );
}
