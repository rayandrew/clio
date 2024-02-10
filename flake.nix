{
  description = "CLIO";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = { self, flake-parts, ... }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];
      perSystem = { config, self', inputs', pkgs, system, ... }:
        let
        env = if system == "x86_64-linux" then
        pkgs.buildFHSUserEnv {
          name = "clio-env";

          targetPkgs = _: [
            pkgs.micromamba
          ];

          profile = ''
            set -e
            eval "$(micromamba shell hook --shell=posix)"
            export MAMBA_ROOT_PREFIX=${builtins.getEnv "PWD"}/.mamba
            micromamba create -q --name clio --file env.yaml
            micromamba activate clio
            set +e
          '';
        }.env else pkgs.mkShell {
          name = "clio-env";
          buildInputs = [ pkgs.micromamba ];
          shellHook = ''
            PWD=$(git rev-parse --show-toplevel)
            eval "$(micromamba shell hook --shell=posix)"
            export MAMBA_ROOT_PREFIX=$PWD/.mamba
            # check environment already exists
            # check if env.yaml has been updated
            if [ ! -d $MAMBA_ROOT_PREFIX/envs/clio ]; then
              echo -e "\033[0;34mCreating environment\033[0m"
              rm -rf $MAMBA_ROOT_PREFIX/envs/clio
              micromamba create -q -y --name clio --file env.yaml
              # save last date of environment creation
              date > $MAMBA_ROOT_PREFIX/envs/clio/.created
            fi
            if [ -f $MAMBA_ROOT_PREFIX/envs/clio/.created ] && [ env.yaml -nt $MAMBA_ROOT_PREFIX/envs/clio/.created ]; then
              echo -e "\033[0;34mCreating environment\033[0m"
              rm -rf $MAMBA_ROOT_PREFIX/envs/clio
              micromamba create -q -y --name clio --file env.yaml

              # echo -e "\033[0;34mUpdating environment\033[0m"
              # micromamba env update -q -y --name clio --file env.yaml
# save last date of environment creation
              date > $MAMBA_ROOT_PREFIX/envs/clio/.created
            fi
            echo -e "\033[0;34mActivating environment\033[0m"
            micromamba activate clio
          '';
        };
      in
      {
        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              micromambaAlpha = prev.callPackage ./nix/micromamba.nix { };
            })
          ];
          config = { };
        };
        devShells.default = env;
      };
    };
}
