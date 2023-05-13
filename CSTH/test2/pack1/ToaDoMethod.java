package pack1;
import java.util.*;
import java.text.*;

public class ToaDoMethod {
  public static double getDist(ToaDo p1,ToaDo p2) {
    return Math.sqrt( Math.pow((p1.x - p2.x),2) 
    + Math.pow((p1.y - p2.y),2) );
  }
  public static void addRelayNode(ToaDo p1,ToaDo p2,double radius,List<ToaDo> tapRN){
    double dist = getDist(p1, p2);
    for (int i=1;i < Math.ceil(dist/radius);i++){
      if (i % 2 == 1) continue;
      double xk = p1.x + i * radius * (p2.x - p1.x) / dist;
      double yk = p1.y + i * radius * (p2.y - p1.y) / dist;
      ToaDo p3 = new ToaDo(xk,yk);
      //System.out.println(p3.toCircle(radius));
      tapRN.add(p3);
    }
  }
  public static ToaDo getSteinerPoint(ToaDo p1,ToaDo p2,ToaDo p3){
    double w = getDist(p1, p2);
    double v = getDist(p1, p3);
    double u = getDist(p2, p3);
    //---------------------------
    double p = (u + v + w) / 2;
    double S = Math.sqrt(p * (p - u) * (p - v) * (p - w));
    double k = Math.sqrt( (u*u + v*v + w*w) / 2 + 2*Math.sqrt(3) * S );
    double d = (v*v + w*w - 2*u*u + k*k) / (3 * k);
    double e = (u*u + w*w - 2*v*v + k*k) / (3 * k);
    double f = (v*v + u*u - 2*w*w + k*k) / (3 * k);
    //------------------------
    double tuSoX = (f*f - d*d - p3.x*p3.x + p1.x*p1.x - p3.y*p3.y + p1.y*p1.y) * (p1.y - p2.y) - (e*e - d*d - p2.x*p2.x + p1.x*p1.x - p2.y*p2.y + p1.y*p1.y) * (p1.y - p3.y);
    double mauSoX = 2 * ((p1.x - p3.x)*(p1.y - p2.y) - (p1.x - p2.x)*(p1.y - p3.y));
    double sx = tuSoX / mauSoX;
    //--------------------------
    double tuSoY = (f*f - d*d - p3.x*p3.x + p1.x*p1.x - p3.y*p3.y + p1.y*p1.y) * (p1.x - p2.x) - (e*e - d*d - p2.x*p2.x + p1.x*p1.x - p2.y*p2.y + p1.y*p1.y) * (p1.x - p3.x);
    double mauSoY = 2 * ((p1.x - p2.x)*(p1.y - p3.y) - (p1.x - p3.x)*(p1.y - p2.y));
    double sy = tuSoY / mauSoY;
    //--------------------------
    return new ToaDo(sx,sy);
  }
  public static void setSteinerPoint(ToaDo p1,ToaDo p2,ToaDo p3,double radius, List<ToaDo> tapRN){
    ToaDo s = getSteinerPoint(p1, p2, p3);
    //System.out.println(s.toCircle(radius));
    tapRN.add(s);
    addRelayNode(p1, s, radius, tapRN);
    addRelayNode(p2, s, radius, tapRN);
    addRelayNode(p3, s, radius, tapRN);
  }
  public static double numOfRelayByMST(ToaDo p1,ToaDo p2,ToaDo p3,double radius){
    double dist[] = new double[4];
    dist[1] = getDist(p2,p3);
    dist[2] = getDist(p1,p3);
    dist[3] = getDist(p1,p2);
    for (int i=1;i < 4; i++)
    for (int j=i+1;j < 4;j++)
    if (dist[i] > dist[j]) {
      double tmp = dist[i];dist[i] = dist[j];dist[j] = tmp;
    }
    return Math.ceil(dist[1] / radius) + Math.ceil(dist[2] / radius);
  }
  public static double numOfRelayBySMT(ToaDo p1,ToaDo p2,ToaDo p3,double radius){
    ToaDo s = getSteinerPoint(p1, p2, p3);
    double su = getDist(p1,s);
    double sv = getDist(p2,s);
    double sw = getDist(p3,s);
    return Math.ceil(su / radius) + Math.ceil(sv / radius) + Math.ceil(sw / radius);
  }
  public static double gain(ToaDo p1,ToaDo p2,ToaDo p3,double radius) {
    return numOfRelayByMST(p1, p2, p3, radius) - numOfRelayBySMT(p1, p2, p3, radius);
  }
}