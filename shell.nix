{ nixpkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

nixpkgs.mkShell {
  nativeBuildInputs = with nixpkgs; [
     uv
  ];
  shellHook = ''
    export CUDA_PATH=${nixpkgs.cudatoolkit}
    export LD_LIBRARY_PATH=${nixpkgs.stdenv.cc.cc.lib}
  '';
}
