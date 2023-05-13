package pack1;
import java.util.*;
import java.text.*;

public class Cluster{
  private ToaDo centroid;
  private List<ToaDo> pointOfCluster;

  public Cluster(ToaDo center){
    centroid = new ToaDo(center);
    pointOfCluster = new ArrayList<ToaDo>();
  }
  public void add(ToaDo diem){
    pointOfCluster.add(diem);
  }
  public ToaDo center(){
    ToaDo average = new ToaDo(0,0);
    for (ToaDo diem: pointOfCluster){
      average.setX(average.getX() + diem.getX());
      average.setY(average.getY() + diem.getY());
    }
    int pointCount = pointOfCluster.size();
    average.setX(average.getX() / pointCount);
    average.setY(average.getY() / pointCount);
    return average;
  }
  public ToaDo getCentroid(){
    return centroid;
  }
  public List<ToaDo> returnList(){
    return pointOfCluster;
  }
}