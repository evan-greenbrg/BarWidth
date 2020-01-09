# River widths and Bar widths
### An automated method to find the widths of river channels and the widths of channel-bar clinoforms.

### General Workflow
Feed the program: 
    - DEM file
    - CSV of channel centerline (in lat-lon)
    - CSV of bar upstream and downstream coordinates (in lat-lon)
    - Cross-Section length

General workflow of the program:
    1. Smooths river centerline to reduce angularity
    2. Converts centerline Lat-Lon to UTM
    3. Loads DEM into memory
    4. Finds what part of centerline is in DEM
    5. Finds the channel direction, and inverse channel direction (will be used to sample the cross-section points)
    6. Constructs the channel-cross sections at the interval of the channel centerline and at the resolution of the DEM
    7. Smooths the Cross-Section to make it more workable
    8. Finds the channel width
    9. Produces a csv of channel bank positions
    10. Loads in the channel-bar positions
    11. Converts bar positions to UTM
    12. Finds the bars that are found within the DEM
    13. Finds the cross-sections that are found within the channel bars
    14. Finds the channel-bar clinoform widths
    15. Produces a csv of channel bar positions
    16. Produces a JSON with the coordinates, clinoform widths, channel widths

Things that could be improved
    - I could feed in UTM initially so I wouldn't have to convert. This is a relic to how I was initially doing things. Mostly not being throughtful about program design. 
    - Width algorithms are fairly naive. The channel width is taken from the biggest positive and biggest negative differences in elevation between two points. This means the program will only find one channel per cross section, which seems fine for what I want to do with the program. The bar is identified by the bank side that has the most gradual slope. I am doing the hard part of identifying whether there is or is not a bar.  
    - There are a number of smoothing parameters that are fed into the program

