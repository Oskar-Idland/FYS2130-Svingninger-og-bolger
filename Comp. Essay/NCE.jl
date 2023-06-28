using LinearAlgebra
using Plots
using Profile

function calc_forces(r, N, cutoff)
    a = zeros(N,2)
    u = 0
    for j ∈ 1:N
        for k ∈ j+1:N
            if j != k
            rvec = r[j,:] .- r[k,:]
            rvecnorm = norm(rvec, 1)
            rvecnorm2 = rvecnorm^2
                if rvecnorm < cutoff && rvecnorm > 0
                    a[j,:] += 24*(2/rvecnorm2^6  -1/rvecnorm2^3)*rvec/rvecnorm2
                    a[k,:] -= a[j,:]
                    u += 4*(rvecnorm2^(-6) - rvecnorm2^(-3))                    
                end
            end
        end
    end
    return a, u
end

function integrate(r, v, a , dt, N, cutoff)
    aprev = a
    r += v*dt + 0.5*a*dt^2
    a, u = calc_forces(r, N, cutoff)
    v += 0.5*(a+aprev)*dt
    return r, v, a, u    
end


function run(dt, N, steps, r, v)
    a = zeros(steps, N, 2)
    u = zeros(steps)
    cutoff = 3
    for i in 1:steps-1
        r[i+1,:,:], v[i+1,:,:], a[i+1,:,:], u[i+1] = integrate(r[i,:,:], v[i,:,:], a[i,:,:], dt, N, cutoff)
    end
    return r, v
end

function scatter_particles(N,l)
    lb = -l*0.5
    ub = -lb
    pos = rand(lb:l/100:ub, N, 2)
    vel = rand(0.1:0.001:sqrt(6), N, 2)
    return pos, vel
end

using Random

function initial_particles(N, l, u)
    # Generate N particles randomly in a box with side lengths l
    # ensuring minimum distance u between each point

    # Initialize an empty array to store the particles
    particles = zeros(N, 2)
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
    
    velocities = rand(0.1:0.001:sqrt(6), N, 2)

    return particles, velocities
end


function SoundWave(N, l, u, t, dt)
    steps = Int(t/dt)
    r_0, v_0 = initial_particles(N, l, u)
    r = zeros(steps, N, 2)
    v = zeros(steps, N, 2)
    r[1,:,:] = r_0
    v[1,:,:] = v_0
    return r, v
end 

N = 300
l = 20
t = 0.25
u = 0.0001
dt = 5*10^(-3)
steps = Int(t/dt)
r, v = SoundWave(N,l,u,t,dt)

@timed r, v = run(dt, N, steps, r, v)


@gif for i ∈ 1:steps
    rx = r[i,:,1]
    ry = r[i,:,2]
    scatter(rx,ry, xlim=(-l,l), ylim=(-l,l), markersize=1, legend=false)
end 
display()