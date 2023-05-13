package pack1;
import java.util.*;
import java.text.*;

public class Sensor {
  private static DecimalFormat df;
  public ToaDo center;
  public double radius;
  private List<Integer> targetCovered;
  private int INVALID = -1;

  public Sensor(ToaDo tam,double rad){
    setCenter(tam);
    radius = rad;
  }
  public void setCenter(ToaDo tam){
    center = new ToaDo(tam);
    targetCovered = new ArrayList<Integer>();
  }
  public String toCircle(){
    df = new DecimalFormat("#.###");
    return "(x-" + df.format(center.getX()) +")^2 +(y-" + df.format(center.getY()) + ")^2 = " + df.format(radius * radius / 4) ;
  }
  public int getTargetCount(){
    return targetCovered.size();
  }
  public int isHave(int target){
  /*
    Tra ve INVALID neu khong co
    Tra ve index neu co
  */
    for (int i=0;i < targetCovered.size();i++)
      if (targetCovered.get(i) == target) return i;
    return INVALID;
  }
  public void add(int target){
    int id = isHave(target);
    if (id == INVALID) 
      targetCovered.add(target);
  }
  public void remove(int target){
    int id = isHave(target);
    if (id != INVALID) targetCovered.remove(id);
  }
  public boolean isCover(ToaDo target){
    double EPS = (double) 1e-10;
    if (center.khoangCach(target) - radius < EPS) return true;
    return false;
  }
  public Integer[] returnList(){
    Integer[] list = new Integer[targetCovered.size()];
    for (int i=0;i < targetCovered.size();i++)
      list[i] = targetCovered.get(i);
    return list;
  }
}