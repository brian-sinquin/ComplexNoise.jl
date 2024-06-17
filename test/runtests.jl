using ComplexNoise
using Test
using Statistics
using FFTW
using DSP
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


function calc_psd(X, NFFT, Fs)


	dt = 1 / Fs
	df = 1 / NFFT / dt
	h = 1.0
	k = length(X) ÷ (NFFT)
	index = 1:NFFT
	norm(x) = 1
	KMU = k * norm(h)^2
	P = zeros(NFFT)
	for i in 1:k
		xw = h .* X[index]
		index = index .+ NFFT
		P = P .+ abs.(fft(xw)) .^ 2
	end
	Pxx = dt * (P[2:NFFT÷2+1]) / KMU / NFFT
	n = length(Pxx)
	F = (0:n-1) * df
	return Pxx, F

end

df = 1
dσ = 1e-10
N = round(Int, 2^16)
fs = (1:1:N) * df
xpsd = dσ ./ fs .^ 2 .+ 1e-5 * dσ .* exp.(-(10 * (fs .- 1.5e4) / 10000) .^ 2) .* (1e4 .< fs .<= 3e4)
_, s = ComplexNoise.PSDToTime(xpsd, df)

dsp = DSP.welch_pgram(s, 64 * 1024, 4, fs = 2 * N * df, window = hanning)

plot(dsp.freq, 10log10.(dsp.power))
plot!(fs, 10log10.(xpsd), lw = 2, color = :black)
xlims!(0, 50)
dsp.power
