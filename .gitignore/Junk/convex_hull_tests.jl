f1(x) = x^2
f2(x) = abs(x)

xs = range(-2, 2, length=200)
pts_f1 = [(x, f1(x)) for x in xs]
pts_f2 = [(x, f2(x)) for x in xs]

using GeometryBasics, QHull

all_pts = vcat(pts_f1, pts_f2)
qhull_input = [Point2f0(p...) for p in all_pts]
hull = chull(qhull_input)

using LinearAlgebra

function is_lower_edge(p1, p2)
    return p2[1] > p1[1]
end

# Convert indices into points
hull_pts = qhull_input[hull]

# Sort hull points by x to walk the boundary
sorted_hull_pts = sort(hull_pts, by = p -> p[1])

# Extract the lower envelope
lower_env_pts = [sorted_hull_pts[1]]
for i in 2:length(sorted_hull_pts)
    if sorted_hull_pts[i][2] < lower_env_pts[end][2]  # y-value drops
        push!(lower_env_pts, sorted_hull_pts[i])
    end
end