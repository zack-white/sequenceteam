{
    c[$1]++
    conf[NR]=$3
    sum_conf+=$3
    sumsq_conf+=$3*$3
}
END{
    for(p in c){
        vals[n++]=c[p]
        sum_labels+=c[p]
        if(c[p]>max_labels) max_labels=c[p]
    }

    mean_labels=sum_labels/n
    for(i=0;i<n;i++){sq+=(vals[i]-mean_labels)^2}
    std_labels=sqrt(sq/n)

    asort(conf)

    mean_conf=sum_conf/NR
    std_conf=sqrt((sumsq_conf/NR)-(mean_conf^2))

    min_conf=conf[1]
    max_conf=conf[NR]

    if(NR%2){
        median=conf[(NR+1)/2]
    } else {
        median=(conf[NR/2]+conf[NR/2+1])/2
    }

    q1=conf[int(NR*0.25)]
    q3=conf[int(NR*0.75)]

    print "Filename:",fname,"\n"

    print "Proteins:",n
    print "Mean GO labels:",mean_labels
    print "Std:",std_labels
    print "Max:",max_labels

    print "\nConfidence Scores:"
    print "Count:",NR
    print "Mean:",mean_conf
    print "Std:",std_conf
    print "Min:",min_conf
    print "Q1:",q1
    print "Median:",median
    print "Q3:",q3
    print "Max:",max_conf
}
