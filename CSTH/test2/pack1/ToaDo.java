package pack1;
import java.text.*;

public class ToaDo{
  private static DecimalFormat df;
  public double x,y;
  public ToaDo(ToaDo other){
    this(other.getX(),other.getY());
  }
  public ToaDo(double ax,double ay){
    setX(ax);
    setY(ay);
  }
  public double getX(){
    return x;
  }
  public double getY(){
    return y;
  }
  public void setX(double ax){
    x = ax;
  }
  public void setY(double ay){
    y = ay;
  }
  public boolean equal(ToaDo other){
    double EPS = (double) 1e-10;
    if (Math.abs(x-other.x) < EPS && Math.abs(y - other.y) < EPS) return true;
    return false;
  }
  public double khoangCach(ToaDo other){
    return Math.sqrt( Math.pow((x - other.x),2) 
    + Math.pow((y - other.y),2) );
  }
  public String toDecimalString(){
    df = new DecimalFormat("#.###");
    return "(" + df.format(x) + "," + df.format(y) + ")";
  }
}