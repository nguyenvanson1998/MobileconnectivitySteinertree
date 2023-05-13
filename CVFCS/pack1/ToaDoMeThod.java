package pack1;
import java.util.*;
import java.text.*;

public class ToaDoMeThod{
  public static double EPS = 10e-7;
  public static double EPS2 = 10e-9;
  public static double cos120 = -0.5;
  public ToaDo nearestToaDo(ToaDo a,ArrayList<ToaDo> List) {
	  ToaDo nearest = List.get(0);
	  for(ToaDo b: List) {
		  if(khoangCach(a,b) < khoangCach(a,nearest))
			  nearest = b;
	  }
	  return nearest;
  }
  public ToaDo trungDiem(ToaDo a,ToaDo b){
    double x = (a.x + b.x) /2;
    double y = (a.y + b.y) /2;
    return new ToaDo(x,y);
  }
  public double khoangCach(ToaDo a,ToaDo b){
    return Math.sqrt( Math.pow((a.x - b.x),2) 
    + Math.pow((a.y - b.y),2) );
  }
  public double heSoA(ToaDo a,ToaDo b){
  // duong thang y = ax + b
  // return he so a
    if (Math.abs(a.x - b.x) < EPS2)
      return 9999;// TH duong thang co dang x = m
    else
      return (a.y - b.y) / (a.x - b.x);
  }
  public double heSoB(ToaDo a,ToaDo b){
  // duong thang y = ax + b
  // return he so b
    return a.y - heSoA(a,b) * a.x;
  }
  public double phiaNaoDuongThang(double a,double b,ToaDo point){
  // diem point nam o phia nao cua duong thang y = ax + b
    return a * point.x - point.y + b;
  }
  public ToaDo dinhThuBaDacBiet_ver1(ToaDo a,ToaDo b,ToaDo diemKhacPhia){
  // return TH dac biet thu 1 cua ham dinhThuBaCuaTamGiac()
    ToaDo trDiem = trungDiem(a,b);
    double canhTamGiac = khoangCach(a,b);
    double hangSo = Math.sqrt(3) * canhTamGiac / 2;
    double x1 = trDiem.x - hangSo;
    double x2 = trDiem.x + hangSo;
    double y = trDiem.y;
    if (diemKhacPhia.x > a.x)
      return new ToaDo(x1,y);
    else
      return new ToaDo(x2,y);
  }
  public ToaDo dinhThuBaDacBiet_ver2(ToaDo a,ToaDo b,ToaDo diemKhacPhia){
  // return TH dac biet thu 2 cua ham dinhThuBaCuaTamGiac()
    ToaDo trDiem = trungDiem(a,b);
    double canhTamGiac = khoangCach(a,b);
    double hangSo = Math.sqrt(3) * canhTamGiac / 2;
    double y1 = trDiem.y - hangSo;
    double y2 = trDiem.y + hangSo;
    double x = trDiem.x;
    if (diemKhacPhia.y > a.y)
      return new ToaDo(x,y1);
    else
      return new ToaDo(x,y2);
  }
  public ToaDo dinhThuBaCuaTamGiac(ToaDo a,ToaDo b,ToaDo diemKhacPhia){
  // return diem thu 3 cua tam giac deu co 2 dinh la a va b
  // nam khac phia doi vs diemKhacPhia
    ToaDo trDiem = trungDiem(a,b);
    double hsA = heSoA(a,b);
    // xu li thuong hop duong thang co dang x = m
    if (Math.abs(hsA - 9999) < EPS2)
      return dinhThuBaDacBiet_ver1(a,b,diemKhacPhia);
    //----------------------------------------------
    // xu li thuong hop duong thang co dang y = m
    if (Math.abs(hsA - 0) < EPS2)
      return dinhThuBaDacBiet_ver2(a,b,diemKhacPhia);
    //----------------------------------------------
    double hsB = heSoB(a,b);
    double hsA_TrungTruc = -1 / hsA;
    double hsB_TrungTruc = trDiem.y - hsA_TrungTruc * trDiem.x;
    double canhTamGiac = khoangCach(a,b);
    // tim ra 2 diem (x1,y1) va (x2,y2)
    double hangSo = Math.sqrt(3) * canhTamGiac / (2 * Math.sqrt(1 + Math.pow(hsA_TrungTruc,2) ));
    double x1 = + hangSo + trDiem.x;
    double y1 = hsA_TrungTruc * x1 + hsB_TrungTruc;
    ToaDo diem1 = new ToaDo(x1,y1);

    double x2 = - hangSo + trDiem.x;
    double y2 = hsA_TrungTruc * x2 + hsB_TrungTruc;
    ToaDo diem2 = new ToaDo(x2,y2);
    // return diem nao khac phia vs diemKhacPhia
    if (phiaNaoDuongThang(hsA,hsB,diemKhacPhia) * phiaNaoDuongThang(hsA,hsB,diem1) > 0) 
      return diem2;
    else 
      return diem1;
  }
  public ToaDo giaoDiemDacBiet(ToaDo a,ToaDo b,ToaDo c,ToaDo d){
  //return ket qua ham giaoDiem() thuong hop duong thang di qua a va b co dang x = m
    ToaDo trDiem = trungDiem(a,b);
    double hsA2 = heSoA(c,d);
    double hsB2 = heSoB(c,d);
    return new ToaDo(trDiem.x, hsA2 * trDiem.x + hsB2);
  }
  public ToaDo giaoDiem(ToaDo a,ToaDo b,ToaDo c,ToaDo d){
  // return giao diem cua 2 duong thang
  // y = ax + b va y = cx + d
    double hsA1 = heSoA(a,b);
    double hsB1 = heSoB(a,b);
    // xu li thuong hop duong thang di qua a va b co dang x = m
    if (Math.abs(hsA1 - 9999) < EPS2)
      return giaoDiemDacBiet(a,b,c,d);
    //------------------------------------------
    double hsA2 = heSoA(c,d);
    double hsB2 = heSoB(c,d);
    // xu li thuong hop duong thang di qua c va d co dang x = m
    if (Math.abs(hsA2 - 9999) <EPS2)
      return giaoDiemDacBiet(c,d,a,b);
    //------------------------------------------
    double x = (hsB2 - hsB1) / (hsA1 - hsA2);
    double y = x * hsA1 + hsB1;
    return new ToaDo(x,y);
  }
  public ToaDo diemSteiner(ToaDo d1,ToaDo d2,ToaDo d3){
    ToaDo d4 = dinhThuBaCuaTamGiac(d1,d2,d3);
    ToaDo d5 = dinhThuBaCuaTamGiac(d1,d3,d2);
    return giaoDiem(d4,d3,d5,d2);
  }
  public double cosGocGiua(ToaDo a,ToaDo b,ToaDo c){
  // return cos goc giua 2 duong thang
  // duong thang 1 di qua a va b
  // duong thang 2 di qua b va c
    double AB = khoangCach(a,b);
    double BC = khoangCach(b,c);
    double AC = khoangCach(a,c);
    return (AB * AB + BC * BC - AC * AC) / (2 * AB * BC);
  }
  public boolean lessThan120(ToaDo a,ToaDo b,ToaDo c){
  // return goc giua 2 duong thang co nho hon 120 do ko ?
  // duong thang 1 di qua a va b
  // duong thang 2 di qua b va c
    double cosGoc = cosGocGiua(a,b,c);
    if ((cosGoc > cos120) && (cosGoc - cos120 > EPS))
      return true;
    else 
      return false;
  }
}