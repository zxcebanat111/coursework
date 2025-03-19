
function model=CreateModelRealData()


    data = importdata('world.txt');

    x=data(:, 2);
    
    y=data(:, 3);
    
    n=numel(x);
    
    d=zeros(n,n);
    
    for i=1:n-1
        for j=i+1:n
            d(i,j)=sqrt((x(i)-x(j))^2+(y(i)-y(j))^2);
            d(j,i)=d(i,j);
        end
    end

    xmin=-60;
    xmax=80;
    
    ymin=-200;
    ymax=200;
    
    model.n=n;
    model.x=x;
    model.y=y;
    model.d=d;
    model.xmin=xmin;
    model.xmax=xmax;
    model.ymin=ymin;
    model.ymax=ymax;
    
end