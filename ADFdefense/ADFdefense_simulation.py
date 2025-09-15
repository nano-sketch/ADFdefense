"""
air defense simulation program AC1, AC2, AC3 are attacking and entering the radar, this shows how they can be intercepted and how long it takes.
the radar has a [ECM] jammer to reduce detection probability, and a launcher with limited missiles and reload time.
so overall this can be used for analysis of radar performance, jamming and enemy aircrafr/missiles attack profiles.  
"""


import math,random,os,datetime,matplotlib.pyplot as plt,numpy as np,pandas as pd
random.seed(1)
import regex as re
#regex is used to extract numbers from strings for plotting hits/misses
#aircraft class for position, speed, heading, radar cross section, alive status and track history.
class Aircraft:
    def __init__(self,i,pos,spd,h,rcs=1.0): self.id=i;self.pos=list(pos);self.spd=spd;self.h=h;self.rcs=rcs;self.alive=True;self.trk=[]
    def step(self,dt):
        if not self.alive:return
        self.pos[0]+=math.cos(self.h)*self.spd*dt;self.pos[1]+=math.sin(self.h)*self.spd*dt;self.trk.append(tuple(self.pos))
#jammer class for position and power.
class Jammer:
    def __init__(self,pos,p): self.pos=tuple(pos);self.p=p
    def eff(self,a,r): d=math.hypot(self.pos[0]-a.pos[0],self.pos[1]-a.pos[1]);k=0.02;return self.p/(self.p+k*(d**2)/max(1e-6,a.rcs))

#radar class for position, range and base detection probability.
class Radar:
    def __init__(self,pos,r,base=0.98): self.pos=tuple(pos);self.r=r;self.base=base
    def det(self,a,js):
        d=math.hypot(self.pos[0]-a.pos[0],self.pos[1]-a.pos[1])
        if d>self.r:return False,d
        jam=0
        for j in js: jam=1-(1-jam)*(1-j.eff(a,self))
        p=self.base*(1-max(0,min(.99,jam)))*(1-d/self.r)
        return random.random()<p,d

#missile class for position, speed, heading, target, number of warheads, max turn rate and range.
class Missile:
    def __init__(self,pos,v,h,t,n=3,mT=math.radians(15),rng=120):
        self.pos=list(pos);self.v=v;self.h=h;self.t=t;self.n=n;self.mT=mT;self.rng=rng
        self.alive=True;self.trk=[];self.start=list(pos);self.prev=None
    def step(self,dt):
        if not self.alive or not self.t.alive:self.alive=False;return
        los=math.atan2(self.t.pos[1]-self.pos[1],self.t.pos[0]-self.pos[0])
        if self.prev is None:self.prev=los
        rate=((los-self.prev+math.pi)%(2*math.pi))-math.pi;self.prev=los
        turn=self.n*rate;turn=max(-self.mT*dt,min(self.mT*dt,turn));self.h=(self.h+turn)%(2*math.pi)
        self.pos[0]+=math.cos(self.h)*self.v*dt;self.pos[1]+=math.sin(self.h)*self.v*dt;self.trk.append(tuple(self.pos))
        if math.hypot(self.pos[0]-self.start[0],self.pos[1]-self.start[1])>self.rng:self.alive=False
#launcher class

class Launcher:
    def __init__(self,pos,n,rl,v,rng): self.pos=tuple(pos);self.n=n;self.rl=rl;self.v=v;self.rng=rng;self.last=-1e9;self.miss=[]
    def canL(self,t): return self.n>0 and t-self.last>=self.rl
    def launch(self,a,t):
        if not self.canL(t):return None
        self.n-=1;self.last=t;h=math.atan2(a.pos[1]-self.pos[1],a.pos[0]-self.pos[0])
        m=Missile(self.pos,self.v,h,a,rng=self.rng);self.miss.append(m);return m

#main simulation func for duration(s), time(s) and save screenshot option [if statement.]
def runSim(dur=400,dt=.5,save=True):
    r=Radar((0,0),120,.99);j=Jammer((80,0),150);l=Launcher((0,0),4,6,1.05,110)
    ac=[Aircraft(1,(150,10),.27,math.pi,1.5),Aircraft(2,(160,-20),.24,math.pi+.04,1.2),Aircraft(3,(200,40),.22,math.pi-.07,.8)]
    t=0;ev=[];allMiss=[]
    while t<dur:
        for a in ac:a.step(dt)
        dets=[]
        for a in ac:
            if not a.alive:continue
            d,rng=r.det(a,[j])
            if d:dets.append((a,rng));ev.append((t,f"det AC{a.id} r={rng:.1f} jam={j.eff(a,r):.2f}"))
        dets.sort(key=lambda x:x[1])
        for a,rng in dets:
            if not l.canL(t):break
            if rng<=l.rng:m=l.launch(a,t)
            if m:allMiss.append(m);ev.append((t,f"launch->AC{a.id} r={rng:.1f} left={l.n}"))
        for m in list(l.miss):
            if not m.alive:l.miss.remove(m);continue
            m.step(dt)
            if math.hypot(m.pos[0]-m.t.pos[0],m.pos[1]-m.t.pos[1])<1:
                p=max(.05,min(.98,.6*((m.v+m.t.spd)/1.2)))
                if random.random()<p:ev.append((t,f"hit AC{m.t.id} at {m.pos[0]:.1f},{m.pos[1]:.1f}"));m.t.alive=False
                else:ev.append((t,f"miss AC{m.t.id} near {m.pos[0]:.1f},{m.pos[1]:.1f}"))
                m.alive=False;l.miss.remove(m)
        t+=dt

    fig,ax=plt.subplots(figsize=(10,8)) #plotting section
    for a in ac:
        tr=np.array(a.trk)
        if tr.size:
            style='-' if a.alive else '--'
            ax.plot(tr[:,0],tr[:,1],style,label=f"AC{a.id}{' down' if not a.alive else ''}")
            for k in range(0,len(tr),int(50/dt)):
                ax.text(tr[k,0],tr[k,1],f"{int(k*dt)}s",fontsize=7,color='gray')
    for m in allMiss:
        tr=np.array(m.trk)
        if tr.size: ax.plot(tr[:,0],tr[:,1],':',alpha=0.7)


    for e in ev:
        if "hit" in e[1] or "miss" in e[1]:
            nums=re.findall(r"[-+]?\d*\.\d+|\d+",e[1])
            if len(nums)>=2:
                x=float(nums[-2]);y=float(nums[-1])
                ax.scatter(x,y,c='red',marker='x',s=60)
    ax.add_artist(plt.Circle(l.pos,r.r,fill=False,ls='--',alpha=0.5))
    ax.scatter([l.pos[0]],[l.pos[1]],marker='^',c='k',s=80)
    ax.set_xlim(-20,210);ax.set_ylim(-90,120)
    ax.grid(True,ls='--',alpha=0.5);ax.set_title("ADFdefense Simulation");ax.legend()

#this checks if the screenshots folder exists in the dir if not it creates one and generates a screenshot.
    if save:
        os.makedirs("screenshots",exist_ok=True)
        fn=datetime.datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
        path=os.path.join("screenshots",fn);plt.savefig(path,dpi=200,bbox_inches='tight');print("saved",path)
    plt.show()
    return ev

if __name__=="__main__": runSim()
