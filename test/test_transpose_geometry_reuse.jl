using Test, Tarang, MPI
MPI.Initialized() || MPI.Init()
const _TGR_NP = MPI.Comm_size(MPI.COMM_WORLD)

@testset "Z-Y geometry keeps dimensional conventions" begin
    # Synthetic topology values suffice: geometry selection performs no MPI call.
    for (rx,ry) in ((1,1),(3,1),(1,3),(2,2)), n in (2,3)
        topo = Tarang.Topology2D(rx,ry,0,0,MPI.COMM_SELF,0,ry,MPI.COMM_WORLD,0,rx)
        comm, peers, pack, unpack = Tarang._zy_transpose_geometry(topo,Val(n))
        if n == 3
            @test comm === MPI.COMM_SELF
            @test (peers,pack,unpack) == (ry,3,2)
        elseif ry > 1
            @test comm === MPI.COMM_SELF
            @test (peers,pack,unpack) == (ry,1,2)
        elseif rx > 1
            @test comm === MPI.COMM_WORLD
            @test (peers,pack,unpack) == (rx,2,1)
        else
            @test comm === nothing
            @test (peers,pack,unpack) == (1,2,1)
        end
    end
end

@testset "Uneven 2D transpose geometry ($_TGR_NP ranks)" begin
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    meshes = _TGR_NP == 1 ? ((1,1),) :
             _TGR_NP == 4 ? ((4,1),(1,4),(2,2)) : ((_TGR_NP,1),(1,_TGR_NP))
    # Independent global values distinguish every element and every direction.
    full = ComplexF64[i + 100j + (i-j)*im for i in 1:9,j in 1:11]
    function block(n,p,r)
        q,rem = divrem(n,p)
        first = r*q + min(r,rem) + 1
        first:(first+q+(r<rem)-1)
    end
    for mesh in meshes
        coords=CartesianCoordinates("x","y")
        dist=Distributor(coords;comm=MPI.COMM_WORLD,mesh,dtype=ComplexF64,
                         architecture=CPU(),use_pencil_arrays=false)
        bases=(ComplexFourier(coords[1];size=9),ComplexFourier(coords[2];size=11))
        field=ScalarField(Domain(dist,bases),"geometry")
        tf=TransposableField(field)
        rx,ry=rank%mesh[1],rank÷mesh[1]
        original=full[block(9,mesh[1],rx),block(11,mesh[2],ry)]
        tf.buffers.z_local_data .= original
        expected_y = if mesh[1]>1 && mesh[2]>1
            full[block(9,mesh[1],rx),:]
        elseif mesh[2]>1
            full[block(9,mesh[2],ry),:]
        else
            full[:,block(11,mesh[1],rx)]
        end
        transpose_z_to_y!(tf)
        @test tf.buffers.y_local_data == expected_y
        @test active_layout(tf) == YLocal
        transpose_y_to_z!(tf)
        @test tf.buffers.z_local_data == original
        @test active_layout(tf) == ZLocal
        if mesh[1]>1 && mesh[2]>1
            # True mesh forward is a gather, reverse is local extraction;
            # asynchronous Z-Y remains explicitly unsupported.
            @test_throws ErrorException async_transpose_z_to_y!(tf)
            @test !tf.async_state.in_progress
            transpose_z_to_y!(tf)
            transpose_y_to_x!(tf)
            @test tf.buffers.x_local_data == full[:,block(11,mesh[1],rx)]
            transpose_x_to_y!(tf)
            @test tf.buffers.y_local_data == expected_y
            transpose_y_to_z!(tf)
            @test tf.buffers.z_local_data == original
        else
            send1,recv1=Tarang.get_active_buffers(tf)
            fill!(send1,ComplexF64(-77));fill!(recv1,ComplexF64(-88))
            async_transpose_z_to_y!(tf)
            if _TGR_NP>1
                @test tf.async_state.in_progress
                @test active_layout(tf) == ZLocal
                @test_throws ErrorException transpose_z_to_y!(tf)
            end
            wait_transpose!(tf)
            @test tf.buffers.y_local_data == expected_y
            @test active_layout(tf) == YLocal
            @test !tf.async_state.in_progress
            @test all(==(-77),send1)
            @test all(==(-88),recv1)
            transpose_y_to_z!(tf)
            @test tf.buffers.z_local_data == original
        end
        close(tf);close(dist)
    end
end
