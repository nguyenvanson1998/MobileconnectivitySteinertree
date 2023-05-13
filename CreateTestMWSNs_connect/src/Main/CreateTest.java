package Main;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
public class CreateTest {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
				Random rand = new Random();
				try {
					BufferedWriter out = new 
			BufferedWriter(new FileWriter(new File("test27.inp")));
					out.write(100+ " " + 100+"\r\n");//kich thuoc mien A
					int bankinh = 8;
					int numbercar = 20;//So luong xe
					int sochuki = 36; // so chu ki
					
					// in ra toa do base
					out.write(rand.nextInt(100) + " " + rand.nextInt(100)+"\r\n");
					// in RA so luong xe
					out.write(numbercar+"\r\n");
					out.write(bankinh+"\r\n");
					out.write(sochuki+"\r\n");
					//random ra so diem busstop moi chu ki
					for (int k = 0; k < numbercar*sochuki-1; k++){
						out.write(rand.nextInt(100) + " " + rand.nextInt(100) +"\r\n");
					}
					out.write(rand.nextInt(100)+ " "+ rand.nextInt(100));
					/*out.write(18+ " " + 22 + "\r\n");
					out.write(20+ " " + 20 + "\r\n");
					out.write(23+ " " + 25 + "\r\n");
					
					out.write(22+ " " + 54 + "\r\n");
					out.write(23+ " " + 52 + "\r\n");
					out.write(24+ " " + 55 + "\r\n");
					
					out.write(50+ " " + 52 + "\r\n");
					out.write(53+ " " + 54 + "\r\n");
					out.write(55+ " " + 49 + "\r\n");
					out.write(100+ " " + 100+ "\r\n");*/
					
					
					out.close();
					System.out.println("success");
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				
			}
		}