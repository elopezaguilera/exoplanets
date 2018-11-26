h:{1!flip `n`v`c!("S * *";8 2 20 3 47)0:0N 80#x};dm:{x[0]#1_0^x:"J"$x[([]n:`NAXIS`NAXIS1`NAXIS2)]`v};tr:{lower trim x except"'"}
ft:{t:x{y;exec tr each v from x where n like y}/:("TTYPE*";"TFORM*");f:flip{"JC"$(-1_;-1#)@\:x}each t 1;
 flip (`$t 0)!l{$[1=x;first;flip]y}'(0,-1_sums l)_((1 1 1 1 2 4 4 8;"cbxxhief")@\:"alxbijed"?f[1]where l:1^f 0)1:(prd dm x)#y}
fi:{b:(1 2 4;"xhi")@\:(),8 16 32?"J"$x[`BITPIX]`v;{$[sum x;x#y;()]}[w;first b 1:(b[0]*prd w:dm x)#y]}
readf:{{e:`$tr(d:h x 0)[`XTENSION]`v;(d;)$[`image~e;fi;`bintable~e;ft;`~e;fi;'e][d;(80+w-(80+count x 0)mod w:36*80)_x 1]}each 0N 2#(0,j:asc raze x ss/:("XTENSION= '";"END",77#" "))_x:"c"$read1 x}

/show each'r:f`$":image.fits";

/ fits bintable/image ext
/ https://archive.stsci.edu/tess/ete-6.html#tpflcdv
/ https://archive.stsci.edu/missions/tess/ete-6/tid/00/000/003/900/tess2019128220341-0000000390018807-0016-s_lc.fits
