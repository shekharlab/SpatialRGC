# $1 = "140g_rn3" (run name)
# $2 cellpose_model_name
# $3 = "--max_z" if you want to run max_z
# $4= "-a" if you want to add transcripts to image (not supported for now)
read -p "Segment and stitch the following regions: " regions
for r in $regions
do
	ID=$(sbatch --parsable make_masks.sh $1_rg$r $2 $3)
	if [ -n "$3" ] ; then
		echo "Running maxz"
		sbatch --dependency=afterok:$ID assign.sh $1_rg$r $2 $3 
	else
		echo "Running stitched"
		sbatch --dependency=afterok:$ID stitch_and_assign.sh $1_rg$r $2
	fi
done	