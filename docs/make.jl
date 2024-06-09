using ComplexNoise
using Documenter

DocMeta.setdocmeta!(ComplexNoise, :DocTestSetup, :(using ComplexNoise); recursive = true)

makedocs(;
	modules = [ComplexNoise],
	authors = "Brian SINQUIN <brian.sinquin@gmail.com> and contributors",
	sitename = "ComplexNoise.jl",
	format = Documenter.HTML(;
		canonical = "https://brian-sinquin.github.io/ComplexNoise.jl",
		edit_link = "master",
		assets = String[],
	),
	pages = ["🏡 Home" => "index.md"],
)

deploydocs(; repo = "github.com/brian-sinquin/ComplexNoise.jl", devbranch = "master")
