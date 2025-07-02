{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
            packages = [
            # General packages
              # pkgs.clangd
	      pkgs.typescript-language-server
              # Python packages
              (pkgs.python312.withPackages (python-pkgs: [
              # packages for formatting/ IDE
                python-pkgs.python-lsp-server
		python-pkgs.flake8
              # packages for code
                python-pkgs.gmsh
		#python-pkgs.ipympl
                python-pkgs.matplotlib
		python-pkgs.numpy
		python-pkgs.numba
		python-pkgs.panel
		python-pkgs.pytest
		python-pkgs.pyvista
		python-pkgs.scipy
		python-pkgs.sympy
		python-pkgs.tomli
              ]))
            ];

	  shellHook = ''
            export VIRTUAL_ENV="Go Van Go - CODE"
	  '';
        };
      });
    };
}
