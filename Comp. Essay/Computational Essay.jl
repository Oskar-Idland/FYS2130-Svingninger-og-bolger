function run_sim()
    using PlotlyJS

    mutable struct Particle
        x::Float64
        y::Float64
        vx::Float64
        vy::Float64
    end

    N::Int = 1_000_000

    print("Time to create particles \n")
    @time particles = [Particle(rand(), rand(), rand(), rand()) for _ in 1:N]
    print("\n")

    print("Time to extract position to matrix \n")
    @time positions = [[p.x, p.y] for p in particles]
    print("\n")


    function advance(particles, dt)
        for p in particles
            p.x, p.y = p.vx*dt, p.vy*dt
        end
    end

    print("Time to advance particles with one time step \n")
    @time advance(particles, 0.1)
    print("\n")

    print("Time to plot $N particle positions using PlotlyJS\n")
    @time plot(scattergl(x=getindex.(positions,1), y=getindex.(positions,2),
    mode = "markers"))
    print("\n")
    
    
    
end


run_sim()