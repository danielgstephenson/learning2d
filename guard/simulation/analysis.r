source = 'simulation'
start=0
end=1000
width=10
sourcePath = paste(source,'.csv',sep='')

simData = read.csv(sourcePath)

frame = simData$frame
time = simData$time
a0x = simData$a0_x
a0y = simData$a0_y
b0x = simData$b0_x
b0y = simData$b0_y
a1x = simData$a1_x
a1y = simData$a1_y
b1x = simData$b1_x
b1y = simData$b1_y
g0x = simData$grad_a0_vx
g0y = simData$grad_a0_vy
g1x = simData$grad_a1_vx
g1y = simData$grad_a1_vy
reward = simData$reward
value = simData$value_estimate
act0 = simData$action0
act1 = simData$action1
c0x = simData$c0x[1]
c0y = simData$c0y[1]
c1x = simData$c1x[1]
c1y = simData$c1y[1]
c2x = simData$c2x[1]
c2y = simData$c2y[1]
c3x = simData$c3x[1]
c3y = simData$c3y[1]
cx = c(c0x,c1x,c2x,c3x,c0x)
cy = c(c0y,c1y,c2y,c3y,c0y)
ringDist = sqrt(a1x^2+a1y^2)
charge = simData$charge
par(cex=2,mar=c(3,3,2,2))
plot(time,charge,type='l',ylim=c(0,1))
chargePath = paste(source,'-charge.png',sep='')
dev.print(png,chargePath,width=1000,height=1000)
drawCircle <- function(x, y, radius, fill = FALSE, n_points = 50, ...) {
  theta <- seq(0, 2 * pi, length.out = n_points)
  xs <- x + radius * cos(theta)
  ys <- y + radius * sin(theta)
  if (fill) {
    polygon(xs, ys, border=NA, ...)
  } else {
    lines(xs, ys, ...)
  }
}
par(cex=2,mar=c(0,0,1,0))
tmin = start
tmax = end #max(time)
s = (tmin<=time&time<=tmax) & (simData$life1>0)
times = time[s]
tmax = min(tmax,max(times))
a0cols = sapply(times,function(x)rgb(0,0.5,0.0,(x-tmin)/(tmax-tmin)))
b0cols = sapply(times,function(x)rgb(0,0.9,0.0,(x-tmin)/(tmax-tmin)))
a1cols = sapply(times,function(x)rgb(0,0.0,0.9,(x-tmin)/(tmax-tmin)))
b1cols = sapply(times,function(x)rgb(0,0.5,1.0,(x-tmin)/(tmax-tmin)))
ringColor = rgb(0,0,0)
wallColor = rgb(0,0,0)
life0 = simData$life0[s]
life1 = simData$life1[s]
a0xs = a0x[s]
a0ys = a0y[s]
b0xs = b0x[s]
b0ys = b0y[s]
a1xs = a1x[s]
a1ys = a1y[s]
b1xs = b1x[s]
b1ys = b1y[s]
xmin = min(a0xs,b0xs,a1xs,b1xs) -width
xmax = max(a0xs,b0xs,a1xs,b1xs) +width 
ymin = min(a0ys,b0ys,a1ys,b1ys) -width
ymax = max(a0ys,b0ys,a1ys,b1ys) +width
plot(x=NA,y=NA,xlim=c(xmin,xmax),ylim=c(ymin,ymax),
     xlab='',ylab='',asp=1,axes=FALSE)
title(main=sprintf("Time %.2f - %.2f", min(times), max(times)))
drawCircle(0,0,radius=15,col=ringColor)
lines(cx,cy,col=wallColor,xpd = NA)
points(a0xs,a0ys,col=a0cols,pch=16,cex=0.5)
points(b0xs,b0ys,col=b0cols,pch=16,cex=0.5)
points(a1xs,a1ys,col=a1cols,pch=16,cex=0.5)
points(b1xs,b1ys,col=b1cols,pch=16,cex=0.5)
end = time==max(time[s])
a0xe = a0x[end]
a0ye = a0y[end]
b0xe = b0x[end]
b0ye = b0y[end]
a1xe = a1x[end]
a1ye = a1y[end]
b1xe = b1x[end]
b1ye = b1y[end]
life0e = life0[end]
life1e = life1[end]
a0col = a0cols[length(a0cols)]
b0col = b0cols[length(b0cols)]
a1col = a1cols[length(a1cols)]
b1col = b1cols[length(b1cols)]
if(life0e) {
  drawCircle(a0xe,a0ye,radius=5,col=a0col,lwd=2)
  segments(a0xe,a0ye,b0xe,b0ye,col=b0col,lwd=2)
}
if(life1e) {
  drawCircle(a1xe,a1ye,radius=5,col=a1col,lwd=2)
  segments(a1xe,a1ye,b1xe,b1ye,col=b1col,lwd=2)
}
drawCircle(b0xe,b0ye,radius=10,col=b0col,lwd=2)
drawCircle(b1xe,b1ye,radius=10,col=b1col,lwd=2)
trajectoryPath = paste(source,'.png',sep='')
dev.print(png,trajectoryPath,width=1000,height=1000)

