shape:{-1_count each first scan x}
accuracy:{avg x=y}
confmat:{d:$[b:x~(::);asc distinct y,z;10b];0^(d!(n;n:count d)#0),exec(count each group c2)d by c1 from $[b;;x=]([]c1:y;c2:z)} / confusion matrix. x is the positive class,y should be the actual classes and z the predictions
confdict:{`tp`fn`fp`tn!raze value confmat[x;y;z]} / number of TP,FN,FP,TN.
precision:('[{x[`tp]%sum x`tp`fp};confdict]) / positive predictive value
sensitivity:('[{x[`tp]%sum x`tp`fn};confdict]) / true positive rate
specificity:('[{x[`tn]%sum x`tn`fp};confdict]) / true negative rate

plt:.p.import`matplotlib.pyplot
displayCM:{[cm;classes;title;cmap]
    
    if[cmap~();cmap:plt`:cm.Blues];
    subplots:plt[`:subplots][`figsize pykw 5 5];
    fig:subplots[`:__getitem__][0];
    ax:subplots[`:__getitem__][1];

    ax[`:imshow][cm;`interpolation pykw `nearest;`cmap pykw cmap];
    ax[`:set_title][`label pykw title];
    tickMarks:til count classes;
    ax[`:xaxis.set_ticks][tickMarks];
    ax[`:set_xticklabels][classes];
    ax[`:yaxis.set_ticks][tickMarks];
    ax[`:set_yticklabels][classes];

    thresh:(max raze cm)%2;
    shp:shape cm;
    {[cm;thresh;i;j] plt[`:text][j;i;(string cm[i;j]);`horizontalalignment pykw `center;`color pykw $[thresh<cm[i;j];`white;`black]]}[cm;thresh;;]. 'cross[til shp[0];til shp[1]];
    plt[`:xlabel]["Predicted Label";`fontsize pykw 12];
    plt[`:ylabel]["Actual label";`fontsize pykw 12];
    fig[`:tight_layout][];
    plt[`:show][];

 }
