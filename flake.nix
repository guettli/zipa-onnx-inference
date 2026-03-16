{
  description = "zipa-onnx-inference dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.nodejs_22
            pkgs.nodePackages.pnpm
            pkgs.tsx               # TypeScript runner (used directly in Taskfile)
            pkgs.ffmpeg            # required: audio decoding (WAV, FLAC, MP3, ...)
            pkgs.go-task           # Taskfile runner
          ];
        };
      }
    );
}
