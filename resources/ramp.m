function f=ramp(k,ds,c,a)
f=(c^2/(2*ds))*(a*r(c*pi*k)+((1-a)/2)*r(pi*c*k+pi)+((1-a)/2)*r(pi*c*k-pi));
end

function vec=r(t)
vec=(sin(t)./t)+(cos(t)-1)./(t.^2);
vec(t==0)=0.5;
end