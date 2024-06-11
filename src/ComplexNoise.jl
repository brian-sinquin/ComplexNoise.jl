module ComplexNoise
using FFTW
# Structs 

# abstract type AbstractSignal end

# mutable struct Signal{T1 <: Number, T2 <: Real} <: AbstractSignal
# 	sn::AbstractArray{T1}
# 	fs::T2
# end

# mutable struct Noise{T1 <: Number, T2 <: Real} <: AbstractSignal
# 	sn::AbstractArray{T1}
# 	fs::T2
# end

# mutable struct NoisySignal{T1 <: Number, T2 <: Real} <: AbstractSignal
# 	noise::Noise{T1, T2}
# 	signal::Signal{T1, T2}
# end

# mutable struct PSD{T <: Real}
# 	psd::AbstractArray{T}
# 	f::AbstractArray{T}
# 	scale::Symbol
# 	unit::Symbol
# end

#!TODO define algebric operations between subtypes

# Noise transformation

# PSDToTimeNoise(psd::Function, f::AbstractArray{T}, n::Int, scale = :log10, unit = :std) where T <: Real = @warn "To Implement"
# PSDToTimeNoise(psd::AbstractArray{T}, f::AbstractArray{T}, n::Int, scale = :log10, unit = :std) where T <: Real = @warn "To Implement"

# Amplification 

# amplify(s, g) = @warn "To Implement"
# Noise measurement 

# phaseNoise(s::AbstractArray{T1}, fs::T2) where {T1 <: Number, T2 <: Real}     = @warn "To Implement"
# amplitudeNoise(s::AbstractArray{T1}, fs::T2) where {T1 <: Number, T2 <: Real} = @warn "To Implement"
# intensityNoise(s::AbstractArray{T1}, fs::T2) where {T1 <: Number, T2 <: Real} = @warn "To Implement"
# SNR 
# CNR
# RIN 

# Multiply signals -> frequency multiplication
# Baseband to Carrier transfer
# Noise factor
# Physical noises
# Need for Abstract signal and noise representation that can be computed into numerical signals

# num to compound NoisySignal and vice versa

# Noise generation

# genPinkNoise(σ::T, fs::T, n::Int) where {T <: Real} = @warn "To Implement"
# genWhiteNoise(σ::T, fs::T, n::Int) where {T <: Real} = @warn "To Implement"

# Noise graphical representation
# Constellation
# Fresnel space 
# Welch PSD 
# Spectrum
# Beta line / Linewidth measurement
# Interferometry

# https://adsabs.harvard.edu/full/1995A%26A...300..707T
#https://stackoverflow.com/questions/29095386/compute-time-series-from-psd-python
"""
PSDToTime(psd, fmin)

Computes a time domain sequence for a given PSD (Power Spectral Density). PSD must be a linear unit variance density (`[x]²/Hz`) with equi-distant frequency points.
"""
function PSDToTime(psd::AbstractArray{T1}, fmin::T2) where {T1 <: Real, T2 <: Real}
	#!TODO lower number of operations against math readability ?
	# PSD2TD (PSD, fmin)
	# generate time-domain data for gaussian noise with given PSD
	# input:
	#     PSD : an array of PSD values, equally spaced in frequency, with interval fmin
	#           array has length N
	#     fmin: first frequency (assumed other frequencies to be n*fmin)
	#            not zero!
	# output:
	#     t: time vector
	#     x:  data evenly spaced in time, with interval Ts
	# for input of size N, time data has size 2N+2

	v = psd * fmin / 2 # variance for quadratures
	N = length(psd)

	#!TODO direct complex randn
	vi = randn(N) # variance 1
	vq = randn(N)
	w = (vi .+ 1im * vq) .* sqrt.(v)

	# length of returned time sequence 
	N_return = 2 .* (length(w) .+ 1)
	# sampling time of returned time sequence
	Ts = 1.0 / (N_return * fmin)

	# compute time vector
	x = FFTW.irfft(vcat([0], w, [0]), N_return)
	# add 0 amplitude at 0 frequency
	# add 0 amplitude at semi-alternate value (exact Nyquist frequency)
	x = x * N_return / sqrt(2)
	# correct for inverse fft factor, 
	# and correct for rms values

	t = LinRange(0, Ts * (N_return - 1), N_return)
	# prepare time vector for plots

	return t, x
end

end



