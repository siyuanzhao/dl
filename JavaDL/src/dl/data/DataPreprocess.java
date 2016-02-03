package dl.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class DataPreprocess {

	public static void main(String[] args) {
		preprocessData();
		
	}
	
	public static int getIndex(Set<? extends Object> set, Object value) {
		int result = 0;
		for (Object entry : set) {
			if (entry.equals(value))
				return result;
			result++;
		}
		return -1;
	}
	
	public static void preprocessData() {
		Set<Integer> set = new HashSet<>();
		String csvFile = "/Users/Siyuan/Documents/workspace/dl/raw_data/problem_logs_2015.csv";
		//DataPreprocess.class.getResourceAsStream(filePath);
		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";
		int preStuId = 0;
		List<StudentLevelData> list = null;
		try {
			StudentLevelData sld = null;
			list = new ArrayList<>();
			br = new BufferedReader(new InputStreamReader
					(new FileInputStream(new File(csvFile))));
			int count = 0;
			while ((line = br.readLine()) != null) {
				/*
				if(count == 0) {
					count++;
					continue; //skip the header
				}*/
				// use comma as separator
				String[] columns = line.split(cvsSplitBy);
				int studentId = Integer.valueOf(columns[0]);
				int problemSetId = 0;
				try {
					problemSetId = Integer.valueOf(columns[2]);
					set.add(problemSetId);
				} catch(Exception e) {
					continue;
				}
				float floatCorrectness = Float.valueOf(columns[3]);
				int correctness = 0;
				if(floatCorrectness == 1) {
					correctness = 1;
				}
				if(preStuId == studentId) { // data for the same student
					sld.num += 1;
					sld.problemSets.add(getIndex(set, problemSetId));
					sld.correctness.add(correctness);
					
				} else { // data from different student
					if(sld != null) {
						list.add(sld);
					}
					sld = new StudentLevelData();
					sld.num = 1;
					sld.problemSets.add(getIndex(set, problemSetId));
					sld.correctness.add(correctness);
					preStuId = studentId;
				}

			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		System.out.println("The number of skills is " + set.size());
		//print out the skill mapping
		Iterator<Integer> it = set.iterator();
		
		try {
			FileWriter writer = new FileWriter("/Users/Siyuan/Documents/workspace/dl/JavaDL/skill_mapping.csv");
			//first write down the num of skills in the dataset
			writer.append("The number of skill builders is " + set.size() + "\n");
			int index = 0;
			while(it.hasNext()) {
				int skill_id = it.next();
				writer.append(index + "," + skill_id+"\n");
				index += 1;
			}
			writer.flush();
			writer.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
		// shuffle the list
		long seed = System.nanoTime();
		Collections.shuffle(list, new Random(seed));
		
		int totalSize = list.size();
		int trainSize = totalSize*8/10;
		
		List<StudentLevelData> trainList = list.subList(0, trainSize);
		List<StudentLevelData> testList = list.subList(trainSize, list.size());
		Iterator<StudentLevelData> iter = trainList.iterator();
		// write to files
		try {
			//get file path
			FileWriter writer = new FileWriter("/Users/Siyuan/Documents/workspace/dl/JavaDL/builder_train.csv");
			while(iter.hasNext()) {
				StudentLevelData item = iter.next();
				writer.append(item.num + "\n");
				writer.append(item.problemSets.toString().replace("[", "").replace("]", "")
	            .replace(", ", ","));
				writer.append("\n");
				writer.append(item.correctness.toString().replace("[", "").replace("]", "")
			            .replace(", ", ","));
				writer.append("\n");
				
			}
			writer.flush();
			writer.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
		
		iter = testList.iterator();
		try {
			FileWriter writer = new FileWriter("/Users/Siyuan/Documents/workspace/dl/JavaDL/builder_test.csv");
			while(iter.hasNext()) {
				StudentLevelData item = iter.next();
				writer.append(item.num + "\n");
				writer.append(item.problemSets.toString().replace("[", "").replace("]", "")
	            .replace(", ", ","));
				writer.append("\n");
				writer.append(item.correctness.toString().replace("[", "").replace("]", "")
			            .replace(", ", ","));
				writer.append("\n");
				
			}
			writer.flush();
			writer.close();
		} catch(IOException e) {
			e.printStackTrace();
		}
	}
}
