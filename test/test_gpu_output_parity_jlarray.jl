"""
NetCDF output parity between CPU and a device field, with no GPU hardware.

Output is the one place where device data MUST leave the device: NetCDF reads and
writes host memory only. Every task kind takes a different route to get there --
a raw grid copy, a `scales` resample (transform + pad + transform), a scalar
`postprocess` reduction, a parsed expression task, and a coefficient-layout task
that splits complex data onto a leading real/imag axis. Each route is a separate
chance to write host garbage, a stale buffer, or a plausible zero.

The existing device output coverage (test_gpu_checkpoint_staging.jl) is 1-D,
grid-layout, single-record. This file pins the 2-D, multi-record, multi-task
case by writing the SAME model twice -- once on CPU, once on a JLArray device --
and demanding the two files be bit-identical, then checking both against the
analytic source so that agreeing on a wrong answer still fails.

JLArray provides device-like arrays with no driver. The cuFFT stand-in is a CPU
twin field run through Tarang's own CPU transform chain, which is what makes the
comparison exact rather than approximate.

Uniquely-prefixed names (gop_*) -- the full suite shares the Main namespace.
"""

using Test
using Tarang

const _GOP_OK = try
    @eval using JLArrays
    @eval using GPUArrays
    true
catch err
    @info "JLArrays/GPUArrays unavailable; skipping output parity test" err
    false
end

if !_GOP_OK
    @testset "NetCDF output parity: CPU vs JLArray device" begin
        @test_skip "JLArrays/GPUArrays unavailable"
    end
else
    const _GOP = JLArrays.JLArray
    const _GOP_ARCH = Tarang.GPU(JLArrays.JLBackend())
    # Test-scoped only; JLArray is used by nothing else in the package.
    Tarang.is_gpu_array(::_GOP) = true
    Tarang.architecture(::_GOP) = _GOP_ARCH
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::Array) = _GOP(a)
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::_GOP) = a
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::AbstractArray) = _GOP(Array(a))
    Tarang.copy_to_device(a::AbstractArray, ::_GOP) = _GOP(Array(a))
    Tarang.copy_to_device(a::_GOP, ::_GOP) = copy(a)
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}) = _GOP
    Tarang.array_type(::Tarang.GPU{JLArrays.JLBackend}, T::Type) = _GOP{T}

    # ---- cuFFT stand-in: a CPU twin field transformed by Tarang's CPU chain ----
    # Keyed on `scales` as well as bases and dtype. An output task with
    # `scales != 1` resamples, and changing a twin's scales after it has been
    # transformed reallocates its buffers while FFTW's cached plan still refers to
    # the previous allocation -- `ArgumentError: FFTW plan applied to output with
    # wrong memory alignment`. It reproduced only on Julia 1.10; 1.11 and 1.12
    # happened to hand back a compatible alignment, which is luck, not contract.
    # One twin per scales value is planned once and never resized.
    const _GOP_TWINS = Dict{Any, Any}()
    function gop_twin(field)
        get!(_GOP_TWINS, (objectid(field.bases), field.dtype, field.scales)) do
            cdist = Distributor(field.dist.coordsys; dtype=field.dtype, device=CPU())
            twin = ScalarField(Domain(cdist, field.bases), "gop_twin_" * field.name)
            if twin.scales != field.scales
                twin.current_layout = :c
                Tarang.preset_scales!(twin, field.scales)
            end
            twin
        end
    end
    # Copy `src` into the buffer selected by `getter`/`setter` on `dst`,
    # reallocating (via `make`) only when the shape or eltype differ.
    function gop_copy_into!(getter, setter, make, dst, src)
        buf = getter(dst)
        if buf === nothing || size(buf) != size(src) || eltype(buf) != eltype(src)
            setter(dst, make(src))
        else
            copyto!(buf, src)
        end
    end
    function Tarang._gpu_forward_transform_backend!(::Tarang.GPU{JLArrays.JLBackend},
                                                    field::Tarang.ScalarField)
        twin = gop_twin(field)
        gop_copy_into!(Tarang.get_grid_data, Tarang.set_grid_data!, copy, twin,
                       Array(Tarang.get_grid_data(field)))
        twin.current_layout = :g
        Tarang.forward_transform!(twin)
        gop_copy_into!(Tarang.get_coeff_data, Tarang.set_coeff_data!, x -> _GOP(copy(x)),
                       field, Tarang.get_coeff_data(twin))
        return true
    end
    function Tarang._gpu_backward_transform_backend!(::Tarang.GPU{JLArrays.JLBackend}, field)
        twin = gop_twin(field)
        gop_copy_into!(Tarang.get_coeff_data, Tarang.set_coeff_data!, copy, twin,
                       Array(Tarang.get_coeff_data(field)))
        twin.current_layout = :c
        Tarang.backward_transform!(twin)
        gop_copy_into!(Tarang.get_grid_data, Tarang.set_grid_data!, x -> _GOP(copy(x)),
                       field, Tarang.get_grid_data(twin))
        return true
    end

    const GOP_NX, GOP_NZ = 8, 6
    const GOP_TASKS = ("u_grid", "u_fine", "u_sum", "dudx", "u_coeff")

    # Distinct data per record, so an output that silently reuses the first
    # record (or the last) cannot pass.
    gop_seed(rec) = [sin(2pi * (ix - 1) / GOP_NX) * cos(pi * (iz - 1) / (GOP_NZ - 1)) +
                     0.1 * rec * ix * iz
                     for ix in 1:GOP_NX, iz in 1:GOP_NZ]

    # Write the same three-record run on `arch` and read every variable back.
    function gop_run(arch, dir, label)
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=arch)
        xb = RealFourier(coords["x"]; size=GOP_NX, bounds=(0.0, 2pi))
        zb = ChebyshevT(coords["z"]; size=GOP_NZ, bounds=(0.0, 1.0))
        domain = Domain(dist, (xb, zb))
        u = ScalarField(domain, "u")
        ensure_layout!(u, :g)

        handler = Tarang.NetCDFFileHandler(joinpath(dir, "out_$label"), dist, Dict("u" => u))
        Tarang.add_task!(handler, u; name="u_grid")
        Tarang.add_task!(handler, u; name="u_fine", scales=2)
        Tarang.add_task!(handler, u; name="u_sum", postprocess=data -> sum(data))
        Tarang.add_task!(handler, "∂x(u)"; name="dudx")
        Tarang.add_task!(handler, u; name="u_coeff", layout="c")

        for rec in 1:3
            ensure_layout!(u, :g)
            copyto!(Tarang.get_grid_data(u), gop_seed(rec))
            @test Tarang.process!(handler; iteration=rec, sim_time=0.25rec)
        end
        file = Tarang.current_file(handler)

        # The device field must still be device-resident, holding live values:
        # staging for I/O may not swap in a host array or consume the buffer.
        ensure_layout!(u, :g)
        storage = Tarang.get_grid_data(u)

        # save_field/load_field! travels the same host boundary by a different route.
        written = save_field(u, joinpath(dir, "ck_$label"), "u")
        v = ScalarField(domain, "v")
        ensure_layout!(v, :g)
        load_field!(v, written, "u")

        read_back = Dict{String,Any}(name => Tarang.group_ncread(file, "vars", name)
                                     for name in GOP_TASKS)
        read_back["sim_time"] = Tarang.group_ncread(file, "time", "sim_time")
        return read_back, storage, Tarang.get_grid_data(v)
    end

    @testset "NetCDF output parity: CPU vs JLArray device" begin
        GPUArrays.allowscalar(false)
        dir = mktempdir()
        cpu, cpu_storage, cpu_loaded = gop_run(CPU(), dir, "cpu")
        gpu, gpu_storage, gpu_loaded = gop_run(_GOP_ARCH, dir, "gpu")

        @test cpu_storage isa Array
        @test gpu_storage isa _GOP
        @test gpu_loaded isa _GOP

        # Every column but `dudx` reaches the file through the SAME FFTW calls on
        # both runs -- the device path's transforms are the CPU twin above -- so
        # those must agree bit for bit. `dudx` is the exception: a Fourier
        # derivative takes `evaluate_fourier_derivative_cpu!` (explicit spectral
        # loops) on the host and `evaluate_fourier_derivative_gpu!` (FFT, multiply
        # by (ik)^n, inverse FFT) on a device array. Those are different algorithms
        # and can only agree to rounding, so it is compared to a tight tolerance.
        # Demanding `==` there passed on macOS/1.12 by luck and failed on
        # ubuntu-x64/1.10 by last-ulp differences (~1e-16 relative).
        @testset "$name matches on CPU and device" for name in (GOP_TASKS..., "sim_time")
            @test size(cpu[name]) == size(gpu[name])
            if name == "dudx"
                @test cpu[name] ≈ gpu[name] rtol=1e-12 atol=1e-12
            else
                @test cpu[name] == gpu[name]
            end
        end

        # Agreeing on a wrong answer must still fail: check the analytic source.
        third = gop_seed(3)
        @test cpu["u_grid"][3, :, :] ≈ third
        @test gpu["u_grid"][3, :, :] ≈ third
        @test cpu["u_sum"][3, 1] ≈ sum(third)
        @test gpu["u_sum"][3, 1] ≈ sum(third)
        @test Array(cpu_storage) ≈ third
        @test Array(gpu_storage) ≈ third
        @test Array(cpu_loaded) ≈ third
        @test Array(gpu_loaded) ≈ third

        # Every record is appended, and each holds its own data.
        @test size(cpu["u_grid"]) == (3, GOP_NX, GOP_NZ)
        @test cpu["sim_time"] ≈ [0.25, 0.5, 0.75]
        @test cpu["u_grid"][1, :, :] ≈ gop_seed(1)
        @test cpu["u_grid"][1, :, :] != cpu["u_grid"][3, :, :]

        # Shapes that encode a route: `scales=2` resamples, `layout="c"` splits
        # complex coefficients onto a leading real/imag axis.
        @test size(cpu["u_fine"]) == (3, 2GOP_NX, 2GOP_NZ)
        @test size(cpu["u_coeff"]) == (3, 2, GOP_NX ÷ 2 + 1, GOP_NZ)
        @test size(cpu["dudx"]) == (3, GOP_NX, GOP_NZ)
        @test !all(iszero, cpu["dudx"])
        @test !all(iszero, cpu["u_coeff"])
    end
end
