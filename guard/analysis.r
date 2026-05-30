
setwd(this.path::here())
start=8
end=12
width=5
source = 'simulation/simulation'
sourcePath = paste(source,'.csv',sep='')
simData = read.csv(sourcePath)

graphics.off()
windows()

frame = simData$frame
time = simData$time
a0x = simData$a0x
a0y = simData$a0y
b0x = simData$b0x
b0y = simData$b0y
a1x = simData$a1x
a1y = simData$a1y
b1x = simData$b1x
b1y = simData$b1y
reward = simData$reward
value = simData$value
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
ringDist0 = sqrt(a0x^2+a0y^2)
ringDist1 = sqrt(a1x^2+a1y^2)

par(cex=1,mar=c(5,5,2,2))

plot(time,reward,type='l')
chargePath = paste(source,'-reward.pdf',sep='')
dev.print(pdf,chargePath)

plot(time,value,type='l')
valuePath = paste(source,'-value.pdf',sep='')
dev.print(pdf,valuePath)

# s = 15<ringDist1 & ringDist1<20
# plot(ringDist1[s],value[s])
# distValuePath = paste(source,'-dist-value.pdf',sep='')
# dev.print(pdf,distValuePath)
# plot(ringDist1[s],reward[s])
# rewardValuePath = paste(source,'-dist-reward.pdf',sep='')
# dev.print(pdf,rewardValuePath)

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

tmin = start
tmax = end #max(time)
s = (tmin<=time&time<=tmax)
times = time[s]
tmin = max(tmin,min(times))
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
xmin = min(a0xs,b0xs,a1xs,b1xs) - width
xmax = max(a0xs,b0xs,a1xs,b1xs) + width 
ymin = min(a0ys,b0ys,a1ys,b1ys) - width
ymax = max(a0ys,b0ys,a1ys,b1ys) + width

par(cex=2,mar=c(0,0,1,0))
plot(x=NA,y=NA,xlim=c(xmin,xmax),ylim=c(ymin,ymax),
     xlab='',ylab='',asp=1,axes=FALSE)
title(main=sprintf("Time %.2f - %.2f", min(times), max(times)))
drawCircle(0,0,radius=13,col=ringColor,lwd=3)
lines(cx,cy,col=wallColor,xpd = NA)
points(a0xs,a0ys,col=a0cols,pch=16,cex=0.4)
points(b0xs,b0ys,col=b0cols,pch=16,cex=0.4)
points(a1xs,a1ys,col=a1cols,pch=16,cex=0.4)
points(b1xs,b1ys,col=b1cols,pch=16,cex=0.4)
end = times==max(times)
a0xe = a0xs[end]
a0ye = a0ys[end]
b0xe = b0xs[end]
b0ye = b0ys[end]
a1xe = a1xs[end]
a1ye = a1ys[end]
b1xe = b1xs[end]
b1ye = b1ys[end]
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

trajectoryPath = paste(source,'.pdf',sep='')
dev.print(pdf,trajectoryPath)
graphics.off()

