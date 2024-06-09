module ComplexNoise

# Structs 

abstract type AbstractSignal end

mutable struct Signal{T1 <: Number, T2 <: Real} <: AbstractSignal
	sn::AbstractArray{T1}
	fs::T2
end

mutable struct Noise{T1 <: Number, T2 <: Real} <: AbstractSignal
	sn::AbstractArray{T1}
	fs::T2
end

mutable struct NoisySignal{T1 <: Number, T2 <: Real} <: AbstractSignal
	noise::Noise{T1, T2}
	signal::Signal{T1, T2}
end

mutable struct PSD{T <: Real}
	psd::AbstractArray{T}
	f::AbstractArray{T}
	scale::Symbol
	unit::Symbol
end

#!TODO define algebric operations between subtypes

# Noise transformation

PSDToTimeNoise(psd::Function, f::AbstractArray{T}, n::Int, scale = :log10, unit = :std) where T <: Real = @warn "To Implement"
PSDToTimeNoise(psd::AbstractArray{T}, f::AbstractArray{T}, n::Int, scale = :log10, unit = :std) where T <: Real = @warn "To Implement"


# Noise measurement 

phaseNoise(s::AbstractArray{T1}, fs::T2) where {T1 <: Number, T2 <: Real}     = @warn "To Implement"
amplitudeNoise(s::AbstractArray{T1}, fs::T2) where {T1 <: Number, T2 <: Real} = @warn "To Implement"
intensityNoise(s::AbstractArray{T1}, fs::T2) where {T1 <: Number, T2 <: Real} = @warn "To Implement"

# Noise generation

genPinkNoise(σ::T, fs::T, n::Int) where {T <: Real} = @warn "To Implement"
genWhiteNoise(σ::T, fs::T, n::Int) where {T <: Real} = @warn "To Implement"


end
