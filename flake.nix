{
  description = "CLIO";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
  };

  outputs = {
    self,
    flake-parts,
    rust-overlay,
    ...
  } @ inputs:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux" "aarch64-darwin" "x86_64-darwin"];
      perSystem = {
        config,
        self',
        inputs',
        pkgs,
        system,
        ...
      }: let
        nativeBuildInputs = with pkgs;
          [
            rustToolchain
            pkg-config
            cargo-watch
          ]
          ++ lib.optionals pkgs.stdenv.isDarwin [
            darwin.apple_sdk.frameworks.SystemConfiguration
          ];
        buildInputs = with pkgs; [
          gnumake
          micromamba
          texlive.combined.scheme-full
          openssl
          bash-completion
          zsh-completions
          bashInteractive
          cmake
        ];
        rustToolchain = pkgs.pkgsBuildHost.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        env =
          if system == "x86_64-linux"
          then
            pkgs.buildFHSUserEnv
            {
              inherit nativeBuildInputs buildInputs;
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
            }
            .env
          else
            pkgs.mkShell {
              inherit nativeBuildInputs buildInputs;
              name = "clio-env";
              DIRENV_LOG_FORMAT = "";
              shellHook = ''
                ########################################
                # Micromamba
                ########################################

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

                if [ -n "$ZSH_VERSION" ]; then
                  autoload -Uz compinit
                  compinit
                  bashcompinit
                elif [ -n "$BASH_VERSION" ]; then
                  :
                fi
              '';
            };
      in {
        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {})
            (import rust-overlay)
          ];
          config = {};
        };
        devShells.default = env;
      };
    };
}
