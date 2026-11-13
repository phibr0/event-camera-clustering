"""Prototype: validate foreground proposals against recent unmasked events."""
import csv, itertools, json, struct, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'outputs/ball-detection-v2'
BASE=ROOT.parent/'ball-tracking'
K=np.array(json.loads((REPO / 'calibration.json').read_text())['camera_matrix'])


def frames(path):
    with path.open('rb') as f:
        while header:=f.read(12):
            t,n=struct.unpack('<dI',header)
            yield t,np.frombuffer(f.read(n*16),dtype='<f4').reshape(n,4)


def fit(points, row):
    x,y,w,h=(float(row[k]) for k in ('cx','cy','width','height'))
    extent=max(w,h)
    if extent>400 or float(row['events'])<30: return None
    # A clipped arc can resemble a much smaller circle. Keep it for tracking only.
    if min(x,1280-x,y,720-y)<extent*.5+3: return None
    pad=max(72,extent*1.1+12)
    p=points[(points[:,2]<=4)&(abs(points[:,0]-x)<pad)&(abs(points[:,1]-y)<pad)]
    if len(p)<15:return None
    _,ids=np.unique(np.round(p[:,:2]/1.5).astype(np.int32),axis=0,return_index=True)
    p=p[ids]
    seed=(p[:,3]>0)&(abs(p[:,0]-x)<w*.75+3)&(abs(p[:,1]-y)<h*.75+3)
    p=p.copy();p[:,3]=seed
    fgpoints=p[seed];bgpoints=p[~seed]
    if len(fgpoints)>192:fgpoints=fgpoints[np.linspace(0,len(fgpoints)-1,192).astype(int)]
    if len(bgpoints)>192:bgpoints=bgpoints[np.linspace(0,len(bgpoints)-1,192).astype(int)]
    p=np.concatenate([fgpoints,bgpoints])
    if len(fgpoints)<4:return None
    rng=np.random.default_rng(123)
    ids=rng.integers(len(p),size=(256,3));ids[:,0]=rng.integers(len(fgpoints),size=256)
    triples=p[ids,:2].astype(float)
    a,b,c=triples[:,0],triples[:,1],triples[:,2]
    u,v=b-a,c-a
    den=2*(u[:,0]*v[:,1]-u[:,1]*v[:,0])
    valid=abs(den)>1
    a,u,v,den=a[valid],u[valid],v[valid],den[valid]
    u2=(u*u).sum(axis=1);v2=(v*v).sum(axis=1)
    dx=(u2*v[:,1]-v2*u[:,1])/den
    dy=(u[:,0]*v2-v[:,0]*u2)/den
    centers=a+np.stack([dx,dy],axis=1)
    radii=np.hypot(dx,dy)
    valid=(radii>=3)&(radii<=max(64,extent*1.4+6))&(np.linalg.norm(centers-[x,y],axis=1)<radii+6)
    centers,radii=centers[valid],radii[valid]
    if not len(radii):return None
    delta=p[None,:,:2]-centers[:,None,:]
    residual=abs(np.linalg.norm(delta,axis=2)-radii[:,None])
    tolerance=np.maximum(1.5,.05*radii)
    inside=residual<=tolerance[:,None]
    angles=((np.arctan2(delta[:,:,1],delta[:,:,0])+np.pi)*16/(2*np.pi)).astype(int)%16
    bins=np.zeros((len(radii),16),dtype=bool)
    ii,jj=np.nonzero(inside)
    bins[ii,angles[ii,jj]]=True
    coverage=bins.sum(axis=1)
    support=inside.sum(axis=1)
    fg=(inside*(p[:,3]>0)).sum(axis=1)
    scores=(support+2*fg)*(coverage/16)**4*np.sqrt(radii)
    scores[(coverage<9)|(support<12)|(fg<4)]=-1
    best=scores.argmax()
    if scores[best]<0:return None
    return dict(cx=centers[best,0],cy=centers[best,1],radius=radii[best],coverage=int(coverage[best]),support=int(support[best]),fraction=float(support[best]/len(p)),fg=int(fg[best]),score=float(scores[best]))


def run():
    timings=[]
    for path in sorted(ROOT.glob('*-events.bin')):
        name=path.name.removesuffix('-events.bin')
        with (BASE/f'{name}-components.csv').open() as f:
            groups=[list(r) for _,r in itertools.groupby(csv.DictReader(f),key=lambda r:r['time_s'])]
        output=[];latency=[]
        for (t,points),rows in zip(frames(path),groups):
            assert abs(t-float(rows[0]['time_s']))<1e-6
            begin=time.perf_counter();found=[]
            for row in rows:
                if not row['cx']:continue
                circle=fit(points,row)
                if circle is None:
                    q=dict(row);q.update(coverage=0,support=0,fraction=0,fg=0,score=0)
                    found.append(q)
                    continue
                q=dict(row)
                q.update(circle)
                q.update(width=2*circle['radius'],height=2*circle['radius'])
                ray=np.linalg.solve(K,[circle['cx'],circle['cy'],1.])
                r=np.array([[float(row[f'r{i}{j}']) for j in range(3)]for i in range(3)])
                ray=K@r@ray
                q.update(ref_x=ray[0]/ray[2],ref_y=ray[1]/ray[2])
                found.append(q)
            latency.append((time.perf_counter()-begin)*1000)
            if not found:
                q=dict(rows[0]);q.update(cx='',cy='',rank=-1);found=[q]
            output.extend(found)
        with (ROOT/f'{name}-components.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(groups[0][0])+['radius','coverage','support','fraction','fg','score'])
            writer.writeheader();writer.writerows(output)
        timings.append(dict(recording=name,median_ms=float(np.median(latency)),p95_ms=float(np.percentile(latency,95)),max_ms=max(latency),updates=len(latency),over_10_ms=sum(t>10 for t in latency),over_30_ms=sum(t>30 for t in latency)))
        print(name,'candidates',sum(bool(r['cx'])for r in output),'fit p95 ms',round(float(np.percentile(latency,95)),3),flush=True)

    (ROOT/'refinement-timing.json').write_text(json.dumps(timings,indent=2)+'\n')


def self_check():
    angles=np.linspace(0,2*np.pi,160,endpoint=False)
    ring=np.column_stack([400+20*np.cos(angles),300+20*np.sin(angles),np.ones(160),((angles<np.pi/2)).astype(float)])
    noise=np.random.default_rng(42).uniform([335,235,0,0],[465,365,4,0],size=(100,4))
    row=dict(cx=416,cy=310,width=12,height=25,events=100)
    circle=fit(np.concatenate([ring,noise]),row)
    assert circle and np.hypot(circle['cx']-400,circle['cy']-300)<1 and abs(circle['radius']-20)<1
    line=np.column_stack([np.linspace(370,430,100),np.full(100,300),np.ones((100,2))])
    assert fit(line,dict(cx=400,cy=300,width=60,height=3,events=100)) is None
    assert fit(ring,dict(row,cx=1)) is None

if __name__=='__main__':
    self_check()
    run()
