# Road Steepness Analysis for Ithaca, NY

This project calculates the steepness (grade) of roads in Ithaca, NY using contour lines and road network shapefiles.

## Data Sources

- **Contours**: `cugir-008148/Ithaca2ftContours.shp` - 2-foot interval contour lines with elevation data
- **Roads**: `Ithaca NY Roads/Roads.shp` - road network with 868 road segments

## How It Works

The script `calculate_road_steepness.py` performs the following steps:

1. loads both shapefiles
2. reprojects roads to match the contour coordinate system (EPSG:2261)
3. builds an elevation interpolator from contour lines for accurate elevation sampling
4. samples elevation at 20 points along each road segment
5. calculates steepness metrics:
   - **max grade**: maximum grade between any two sample points
   - **avg grade**: overall grade from start to end of road segment
   - **elevation change**: total elevation change along the road

## Installation

install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

run the analysis:

```bash
python3 calculate_road_steepness.py
```

this generates `road_steepness_results.csv` with steepness data for all roads.

## Results

The analysis found:

- **steepest road**: stewart ave with max grade of 347% (likely a short segment with interpolation artifact)
- **average max grade**: 8.93% across all roads
- **275 roads** have sections with grade > 10%

### Notable Steep Roads (realistic grades, 8-30% range)

Some of the steepest roads in Ithaca (for roads > 100 feet long):

- **hillview pl**: 30.38% max grade, 15.12% average
- **edgecliff pl**: 28.22% max grade, 14.75% average
- **cook st**: 24.68% max grade, 15.93% average
- **columbia st**: 26.86% max grade, 13.64% average
- **ferris pl**: 28.27% max grade, 12.31% average

### Understanding the Results

- grades > 100% are likely from very short road segments or data artifacts
- roads with 10-20% grades are considered very steep (ski slopes are often 20-30%)
- most roads (393 out of 842) have max grades ≤ 5%, which is typical for urban streets

## Output Format

the csv file contains:

- `road_name`: street name
- `objectid`: unique identifier
- `length_feet`: road segment length
- `max_grade_pct`: maximum grade percentage along segment
- `avg_grade_pct`: average grade from start to end
- `elevation_change_ft`: total elevation change
- `min_elevation_ft`: minimum elevation
- `max_elevation_ft`: maximum elevation

## Notes

- grade is calculated as (rise/run) × 100%
- very short road segments may show artificially high grades
- for more realistic results, filter for roads longer than 100-200 feet
- the interpolation method provides smooth elevation estimates between contour lines

