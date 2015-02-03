function g_u = tri_kernel( u,h )
%return the triangle kernel model g(u)
absu = abs(u);
g_u = 1- absu/h;

g_u(g_u < 0)=0;

end

