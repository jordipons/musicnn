{ nixpkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

nixpkgs.mkShell {
  nativeBuildInputs = with nixpkgs; [
    uv
    glib
    zlib
    libGL
    fontconfig
    xorg.libX11
    libxkbcommon
    freetype
    dbus

    qt6.qtbase
    qt6.qtwayland

    xorg.libxcb
    xorg.xcbutil
    xorg.xcbutilwm
    xorg.xcbutilimage
    xorg.xcbutilkeysyms
  ];
  shellHook = ''
    export CUDA_PATH=${nixpkgs.cudatoolkit}
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:${nixpkgs.cudaPackages.cudnn.lib}:${nixpkgs.cudatoolkit.lib}:${nixpkgs.stdenv.cc.cc.lib}:$LD_LIBRARY_PATH
  '';
}
