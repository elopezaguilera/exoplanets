/ requires pydl (pip install pydl)
/ requires fits.q (from pierre)

\l p.q
\l ../utils/fits.q

tessFilenames:{[dir;id]` sv dir,first f where (f:key[dir])like "*",string[id],"*"}
readTessLC:{[fl]select time,flux:pdcsap_flux from readf[fl][1;1]where not null pdcsap_flux}

phaseFoldTime:{[tm;p;t0]mod[tm+hp-t0;p]-hp:.5*p}
genview:{[x;nbin;wbin;tmax]
  bins:(0;wbin)+/:neg[tmax]+(((2*tmax)-wbin)%nbin-1)*til nbin;
  flx%abs min flx-:med flx:med each x[`newflux]where each x[`ftime]within/:bins}

processcurve:{[dir;tce]
  st:.z.t;
  x:readTessLC each tessFilenames[dir]tce`catid;
  x:update ftime:phaseFoldTime[time;tce`tce_period;tce`tce_time0bk],newflux:flux from x;
  lview:genview[x;201;.16*tce`tce_duration;(.5*tce`tce_period)&4*tce`tce_duration];
  gview:genview[x;2001;1%2001;.5*tce`tce_period];
  -1"Processed star (",string[tce`catid],") : Time taken (",string[.z.t-st],")";
  `raw`global`local!(x;gview;lview)}



