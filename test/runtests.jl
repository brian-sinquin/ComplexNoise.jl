using ComplexNoise
using Test
using Statistics

@testset verbose = true "PSDToTime" begin
	df = 1
	dσ = 1e-5
	N = round(Int, 2^14)
	psd = dσ * ones(N)
	_, s = ComplexNoise.PSDToTime(psd, df)

	@testset "Array length" begin
		@test length(s) == 2 * (N + 1)
	end
	@testset "Variance conservation" begin
		@test (std(s)^2 - dσ * df * N) / (dσ * df * N) <= 10 / 100
	end
end


