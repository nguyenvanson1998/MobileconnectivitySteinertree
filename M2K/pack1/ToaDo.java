package pack1;
import java.util.*;
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
  public double khoangCach(ToaDo other){
    return Math.sqrt( Math.pow((x - other.x),2) 
    + Math.pow((y - other.y),2) );
  }
  public String toDecimalString(){
    df = new DecimalFormat("0.000");
    return "(" + df.format(x) + "," + df.format(y) + ")";
  }
  public String toCircle(double banKinh){
    df = new DecimalFormat("0.000");
    return "(x-" + df.format(x) +")^2 +(y-" + df.format(y) + ")^2 = " + df.format(banKinh * banKinh) ;
  }
}