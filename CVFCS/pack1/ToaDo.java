package pack1;
import java.util.*;
import java.text.*;

public class ToaDo{
  private static double EPS = 10e-9;
  private static DecimalFormat df;
  public double x,y;
  public int chuKy;
  
  public ToaDo(ToaDo other){
    this(other.getX(),other.getY());
  }
  public ToaDo(double ax,double ay){
    setX(ax);
    setY(ay);
    chuKy = -1;
  }
  public ToaDo(double ax,double ay,int ck){
    this(ax,ay);
    chuKy = ck;
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
    if (Math.abs(x-other.x) < EPS && Math.abs(y - other.y) < EPS) return true;
    return false;
  }
  public double khoangCach(ToaDo other){
    return Math.sqrt( Math.pow((x - other.x),2) 
    + Math.pow((y - other.y),2) );
  }
  public String toDecimalString(){
    df = new DecimalFormat("0.000");
    return "(" + df.format(x) + "," + df.format(y) + ")";
  }
  public void print(){
    System.out.println(toDecimalString() );
  }
  public void printRelay(double banKinh){
    df = new DecimalFormat("0.000");
    System.out.println("(x-" + df.format(x) +")^2 +(y-" + df.format(y) + ")^2 = " + df.format(banKinh * banKinh)) ;
  }
}