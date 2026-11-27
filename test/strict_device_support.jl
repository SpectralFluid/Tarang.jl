# Strict JLArray device: the CUDA rule plain JLArray does not enforce.
#
# `allowscalar(false)` already makes scalar indexing of a JLArray throw, as for a
# CuArray. What JLArray still accepts silently is a HOST `Array` among a device
# kernel's arguments: CUDA converts only CuArrays at launch (`cudaconvert`,
# CUDA/src/compiler/execution.jl), so a host array reaches GPUCompiler, which
# rejects it ("passing non-bitstype argument"). Inside `with_strict_device(f)` a
# host `Array` reaching a JLArray kernel throws `HostArrayInKernel`; outside it,
# JLArray behaves as usual, so files that knowingly mix host arrays are unaffected.
#
# Include-once: the suite runs every file in one Main module, and the flag must be
# a single global (a per-file copy would leave all but the last-defined flag dead).
# The caller has loaded JLArrays.
if !isdefined(Main, :TarangStrictDevice)
    @eval Main module TarangStrictDevice
        import JLArrays
        const Adapt = JLArrays.Adapt    # JLArrays' own binding: no direct Adapt dependency
        export with_strict_device, HostArrayInKernel

        struct HostArrayInKernel <: Exception
            msg::String
        end
        Base.showerror(io::IO, e::HostArrayInKernel) = print(io, "HostArrayInKernel: ", e.msg)

        const STRICT = Ref(false)

        function Adapt.adapt_storage(::JLArrays.Adaptor, x::Array)
            STRICT[] && throw(HostArrayInKernel(
                "host $(typeof(x)) of size $(size(x)) passed into a device kernel"))
            return x
        end

        function with_strict_device(f)
            previous = STRICT[]
            STRICT[] = true
            try
                return f()
            finally
                STRICT[] = previous
            end
        end
    end
end
using Main.TarangStrictDevice: with_strict_device, HostArrayInKernel
