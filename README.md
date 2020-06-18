# River widths and Bar widths
### An automated method to find the widths of river channels and the widths of channel-bar clinoforms.

### General Workflow
Feed the program:  
    -   DEM file   
    -   CSV of channel centerline (in lat-lon)  
    -   CSV of bar upstream and downstream coordinates (in lat-lon)  
    -   Cross-Section length  

Steps:  
    1. Smooths river centerline to reduce angularity  
    2. Converts centerline Lat-Lon to UTM  
    3. Loads DEM into memory  
    4. Finds what part of centerline is in DEM  
    5. Finds the channel direction, and inverse channel direction (will be used to sample the cross-section points)  
    6. Constructs the channel-cross sections at the interval of the channel centerline and at the resolution of the DEM  
    7. Smooths the Cross-Section to make it more workable  
    8. Finds the channel width from elevation cross-section and ESA water occurence cross section 
    9. Produces a csv of channel bank positions  
    10. Loads in the channel-bar positions  
    11. Converts bar positions to UTM  
    12. Finds the bars that are found within the DEM  
    13. Finds the cross-sections that are found within the channel bars  
    14. Finds the channel-bar clinoform widths - Two methods manual and automatic  
    15. Produces products of channel bar geometry, channel bar slopes

