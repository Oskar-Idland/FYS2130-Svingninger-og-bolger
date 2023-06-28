using LinearAlgebra
using Plots
using StaticArrays
using FastPow
using ProgressMeter

function calc_forces(r, N, max_distance)
    a = zeros(N,2)
    u = 0
    for j ∈ 1:N
        for k ∈ j+1:N
            if j != k
            Δr = r[j,:] .- r[k,:]
            r_norm = norm(Δr, 1)
            r_norm_sq = r_norm^2
                if r_norm < max_distance && r_norm > 0
                    @fastpow a[j,:] += 24*(2/r_norm_sq^6  -1/r_norm_sq^3)*Δr/r_norm_sq
                    a[k,:] -= a[j,:]
                    @fastpow u += 4*(r_norm_sq^(-6) - r_norm_sq^(-3))                    
                end
            end
        end
    end
    return a, u
end

function sum_forces(r, v, a , dt, N, max_distance)
    a_prev = a
    r += v*dt + 0.5*a*dt^2
    a, u = calc_forces(r, N, max_distance)
    v += 0.5*(a+a_prev)*dt
    return r, v, a, u    
end


function time_iteration(dt, N, steps, r, v, max_distance)
    a = zeros(steps, N, 2)
    u = zeros(steps)
    @showprogress for i in 1:steps-1
        r[i+1,:,:], v[i+1,:,:], a[i+1,:,:], u[i+1] = sum_forces(r[i,:,:], v[i,:,:], a[i,:,:], dt, N, max_distance)
    end
    return r, v
end


function initial_particles(N, l, u)
    # Generate N particles randomly in a box with side lengths l
    # ensuring minimum distance u between each point

    # Initialize an empty array to store the particles
    particles = [zeros(N) zeros(N)]
    count = 0

    # Generate random coordinates for each point
    while count < N
        x = rand(-l/2:l/1000:l/2)
        y = rand(-l/2:l/1000:l/2)

        # Check the minimum distance u from each existing point
        valid = true
        for i in 1:count
            if norm([x,y] .- particles[i,:]) < u
                valid = false
                break
            end
        end

        # Add the point to the matrix if it satisfies the minimum distance constraint
        if valid
            count += 1
            particles[count,:] = [x, y]
        end
    end
    
    velocities = rand(0.1:0.001:sqrt(2), N, 2)
    velocities = zeros(N,2)

    return particles, velocities
end


function particle_grid(N, l, u)
    particles = zeros(N,2) 
end

function SoundWave(N, l, u, t, dt)
    steps = Int(t/dt)
    r_0, v_0 = initial_particles(N, l, u)
    r = zeros(steps, N, 2)
    v = zeros(steps, N, 2)
    # print(r_0)
    # print(r)
    r[1,:,:] = r_0
    v[1,:,:] = v_0
    return r, v
end 

# struct Simulation
#     N:: Int
#     l:: Float64
#     t:: Float64
#     u:: Float64
#     dt:: Float64
#     steps::Int = (t/dt)
# end



N = 200
l = 20
t = 0.125
dt = 5*10^(-4)
min_distance = 0.5
max_distance = 3
steps = Int(t/dt)
r_0, v_0 = SoundWave(N,l,min_distance,t,dt)

@time r, v = time_iteration(dt, N, steps, r_0, v_0, max_distance)
println(typeof(r[1,:,:]))
print(size(r))


function Wave(r)
    x = LinRange(0,2π,100)
    return r*sin.(x) , r*cos.(x)
end

function plot_particles(r)
end

@gif for i ∈ 1:steps
    rx = r[i,:,1]
    ry = r[i,:,2]
    Plots.scatter(rx,ry, xlim=(-l/2,l/2), ylim=(-l/2,l/2), markersize=2, legend=false)
    Plots.plot!(Wave(i), xlim=(-l/2,l/2), ylim=(-l/2,l/2))
end

# plot_particles(r)